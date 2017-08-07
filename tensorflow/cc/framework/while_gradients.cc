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

#include "tensorflow/cc/framework/while_gradients.h"

#include "tensorflow/cc/framework/gradients.h"
#include "tensorflow/cc/framework/scope_internal.h"
#include "tensorflow/cc/ops/control_flow_ops_internal.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/while_loop.h"
#include "tensorflow/core/graph/node_builder.h"

namespace tensorflow {
namespace {

using ops::CondGraphBuilderFn;
using ops::BodyGraphBuilderFn;
using ops::BuildWhileLoop;

Output ToOutput(OutputTensor output_tensor) {
  return Output(output_tensor.node, output_tensor.index);
}

std::vector<Output> ToOutputVector(
    const std::vector<OutputTensor>& output_tensors) {
  int n = output_tensors.size();
  std::vector<Output> result(n);
  for (int i = 0; i < n; ++i) result[i] = ToOutput(output_tensors[i]);
  return result;
}

}  // namespace

Status AddForwardLoopCounter(WhileContext* while_ctx, const Scope& scope,
                             Output* count) {
  Output zero = ops::Const(scope, 0, {});

  // Create while loop:
  //   i = 0
  //   while forward loop predicate is true:
  //     ++i

  // Condition function that returns condition output from original while loop
  CondGraphBuilderFn cond_fn = [while_ctx](const Scope& scope,
                                           const std::vector<Output>& inputs,
                                           Output* output) {
    *output = ToOutput(while_ctx->cond_output());
    return Status::OK();
  };

  // Body function that adds one to input
  BodyGraphBuilderFn body_fn = [while_ctx](const Scope& scope,
                                           const std::vector<Output>& inputs,
                                           std::vector<Output>* outputs) {
    DCHECK_EQ(inputs.size(), 1);
    outputs->emplace_back(ops::Add(scope, inputs[0], 1));
    return scope.status();
  };

  std::vector<Output> outputs;
  TF_RETURN_IF_ERROR(BuildWhileLoop(scope, {zero}, cond_fn, body_fn,
                                    while_ctx->frame_name(), false, &outputs));
  *count = outputs[0];
  return Status::OK();
}

Status AddBackPropLoopCounter(WhileContext* while_ctx, Output n,
                              const Scope& scope,
                              Output* backprop_execution_pred) {
  // Create while loop: while n > 0: --n

  // Condition function that returns input > 0
  CondGraphBuilderFn cond_fn = [](const Scope& scope,
                                  const std::vector<Output>& inputs,
                                  Output* output) {
    DCHECK_EQ(inputs.size(), 1);
    *output = ops::Greater(scope, inputs[0], 0);;
    return scope.status();
  };

  // Body function that subtracts one from input
  BodyGraphBuilderFn body_fn = [](const Scope& scope,
                                  const std::vector<Output>& inputs,
                                  std::vector<Output>* outputs) {
    DCHECK_EQ(inputs.size(), 1);
    outputs->emplace_back(ops::Subtract(scope, inputs[0], 1));
    return scope.status();
  };

  std::vector<Output> outputs;
  TF_RETURN_IF_ERROR(BuildWhileLoop(scope, {n}, cond_fn, body_fn,
                                    while_ctx->frame_name(), false, &outputs,
                                    backprop_execution_pred));
  return Status::OK();
}

Status AddWhileGradientLoop(WhileContext* while_ctx,
                            const std::vector<Output>& grad_inputs,
                            Output backprop_execution_pred,
                            const Scope& parent_scope,
                            std::vector<Output>* grad_outputs) {
  DCHECK_EQ(grad_inputs.size(), while_ctx->body_outputs().size());
  DCHECK_EQ(while_ctx->body_inputs().size(),
            while_ctx->body_outputs().size());

  Scope scope = parent_scope.NewSubScope("while");

  // Create while loop: while backprop_execution_pred: while body gradient

  // Condition function that returns 'backprop_execution_pred'
  CondGraphBuilderFn cond_fn = [backprop_execution_pred](
                                   const Scope& scope,
                                   const std::vector<Output>& inputs,
                                   Output* output) {
    *output = backprop_execution_pred;
    return Status::OK();
  };

  // Body function that builds while body gradient subgraph
  BodyGraphBuilderFn body_fn = [while_ctx](const Scope& scope,
                                           const std::vector<Output>& inputs,
                                           std::vector<Output>* outputs) {
    std::vector<Output> body_outputs =
        ToOutputVector(while_ctx->body_outputs());
    std::vector<Output> body_inputs = ToOutputVector(while_ctx->body_inputs());
    return AddSymbolicGradients(scope, body_outputs, body_inputs, inputs,
                                outputs);
  };

  TF_RETURN_IF_ERROR(BuildWhileLoop(scope, grad_inputs, cond_fn, body_fn,
                                    while_ctx->frame_name(), false,
                                    grad_outputs));
  return Status::OK();
}

}  // namespace tensorflow
