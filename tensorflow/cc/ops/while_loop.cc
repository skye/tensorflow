/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/cc/ops/while_loop.h"

#include "tensorflow/cc/framework/scope_internal.h"
#include "tensorflow/cc/ops/control_flow_ops_internal.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/common_runtime/shape_refiner.h"

namespace tensorflow {
namespace ops {

namespace {

// Utility function for converting to internal C++ datatypes
OutputTensor ToOutputTensor(Output output) {
  return OutputTensor(output.node(), output.index());
}

// Utility function for converting to internal C++ datatypes
std::vector<OutputTensor> ToOutputTensors(const std::vector<Output>& outputs) {
  std::vector<OutputTensor> result(outputs.size());
  for (int i = 0; i < outputs.size(); ++i) {
    result[i] = ToOutputTensor(outputs[i]);
  }
  return result;
}

// Utility function for converting to internal C++ datatypes
std::vector<Node*> ToNodes(const std::vector<Output>& outputs) {
  std::vector<Node*> result(outputs.size());
  for (int i = 0; i < outputs.size(); ++i) {
    result[i] = (outputs[i].node());
  }
  return result;
}

}  // namespace

Status BuildWhileLoop(const Scope& scope, const std::vector<Output>& inputs,
                      CondGraphBuilderFn cond, BodyGraphBuilderFn body,
                      const string& frame_name, bool create_while_ctx,
                      OutputList* outputs, Output* cond_output) {
  DCHECK(!inputs.empty());
  DCHECK(outputs != nullptr);
  DCHECK(outputs->empty());

  TF_RETURN_IF_ERROR(scope.status());
  int n = inputs.size();

  std::vector<Output> enter_outputs(n);
  for (int i = 0; i < n; ++i) {
    enter_outputs[i] = internal::Enter(scope, inputs[i], frame_name);
  }

  // The merge nodes accept the while loop's back edges as an input (i.e. the
  // not-yet-created next iteration nodes). Use the underlying NodeBuilder API
  // directly to create an input to the not-yet-created back edge.

  // Manually generate what the NextIteration node names will be.
  TF_RETURN_IF_ERROR(scope.status());
  std::vector<string> next_names(n);
  next_names[0] = strings::StrCat(scope.impl()->name(), "/NextIteration");
  for (int i = 1; i < n; ++i) {
    next_names[i] = strings::StrCat(scope.impl()->name(), "/NextIteration_", i);
  }

  // Use NodeBuilder API to build merge nodes
  TF_RETURN_IF_ERROR(scope.status());
  std::vector<Output> merge_outputs(n);
  for (int i = 0; i < n; ++i) {
    NodeBuilder::NodeOut enter_input(
        enter_outputs[i].node(), enter_outputs[i].index());

    DataType dtype = enter_outputs[i].node()->output_type(0);
    NodeBuilder::NodeOut next_input(next_names[i], 0, dtype);

    std::vector<NodeBuilder::NodeOut> input_list({enter_input, next_input});
    string unique_name = scope.GetUniqueNameForOp("Merge");
    NodeBuilder builder = NodeBuilder(unique_name, "Merge").Input(input_list);
    scope.UpdateBuilder(&builder);

    Node* merge_node;
    TF_RETURN_IF_ERROR(builder.Finalize(scope.graph(), &merge_node));
    TF_RETURN_IF_ERROR(scope.DoShapeInference(merge_node));
    merge_outputs[i] = Output(merge_node, 0);
  }

  TF_RETURN_IF_ERROR(scope.status());
  // The control dependency is for constants in the cond graph
  Scope cond_scope =
      scope.NewSubScope("cond").WithControlDependencies(merge_outputs[0]);
  Output raw_cond_out;
  TF_RETURN_IF_ERROR(cond(cond_scope, merge_outputs, &raw_cond_out));
  if (raw_cond_out.type() != DT_BOOL) {
    return errors::InvalidArgument(
        "BuildWhileLoop: 'cond' argument must return a boolean output, got ",
        DataTypeString(raw_cond_out.type()));
  }
  Output cond_out = LoopCond(scope, raw_cond_out).output;
  if (cond_output != nullptr) *cond_output = cond_out;

  std::vector<Output> switch_trues(n);
  std::vector<Output> switch_falses(n);
  for (int i = 0; i < n; ++i) {
    auto swtch = Switch(scope, merge_outputs[i], cond_out);
    switch_trues[i] = swtch.output_true;
    switch_falses[i] = swtch.output_false;
  }

  TF_RETURN_IF_ERROR(scope.status());
  // The control dependency is for constants in the body graph
  Scope body_scope =
      scope.NewSubScope("body").WithControlDependencies(switch_trues[0]);
  std::vector<Output> body_outputs;
  TF_RETURN_IF_ERROR(body(body_scope, switch_trues, &body_outputs));
  if (body_outputs.size() != n) {
    return errors::InvalidArgument(
        "BuildWhileLoop: 'body' argument expected to return ", n,
        "outputs, got ", body_outputs.size());
  }

  std::vector<Output> next_outputs(n);
  for (int i = 0; i < n; ++i) {
    next_outputs[i] = NextIteration(scope, body_outputs[i]);
    DCHECK_EQ(next_outputs[i].node()->name(), next_names[i]);
  }

  // Create the backedges from the NextIteration nodes to the Merge nodes
  for (int i = 0; i < n; ++i) {
    // TOOD(skye): does this export correctly?
    scope.graph()->AddEdge(next_outputs[i].node(), next_outputs[i].index(),
                           merge_outputs[i].node(), 1);
  }

  outputs->resize(n);
  for (int i = 0; i < n; ++i) {
    (*outputs)[i] = internal::Exit(scope, switch_falses[i]);
  }
  TF_RETURN_IF_ERROR(scope.status());

  if (create_while_ctx) {
    WhileContext* while_ctx;
    TF_RETURN_IF_ERROR(scope.graph()->AddWhileContext(
        frame_name, ToNodes(enter_outputs), ToNodes(*outputs),
        ToOutputTensor(cond_out), ToOutputTensors(switch_trues),
        ToOutputTensors(body_outputs), &while_ctx));

    // Set while_ctx for all exit nodes. We currently don't require knowing the
    // while_ctx for any other nodes.
    for (int i = 0; i < n; ++i) {
      (*outputs)[i].node()->set_while_ctx(while_ctx);
    }
  }
  return Status::OK();
}

}  // namespace ops
}  // namespace tensorflow
