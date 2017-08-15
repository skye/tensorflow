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

#ifndef THIRD_PARTY_TENSORFLOW_CC_FRAMEWORK_WHILE_GRADIENTS_H_
#define THIRD_PARTY_TENSORFLOW_CC_FRAMEWORK_WHILE_GRADIENTS_H_

#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/core/graph/while_context.h"

// Utility functions for constructing while loop gradients

namespace tensorflow {

// Creates a loop that counts the number of iterations performed by the while
// loop associated with `while_ctx`. The returned output yields the iteration
// count.
Status AddForwardLoopCounter(WhileContext* while_ctx, const Scope& scope,
                             Output* count);

// Creates a loop that executes `n` times. The returned output is the boolean
// predicate indicating if the loop is still executing. This is used to drive
// the gradient computation for the while loop associated with `while_ctx`.
Status AddBackPropLoopCounter(WhileContext* while_ctx, Output n,
                              const Scope& scope,
                              Output* backprop_execution_pred);

// Creates the main backprop loop that computes the gradient of the loop
// associated with `while_ctx`. `grad_inputs` are the partial derivatives
// w.r.t. the loop outputs, i.e. the exit nodes. `backprop_execution_pred` is
// the predicate to use for the backprop loop (see AddBackPropLoopCounter()).
// The partial derivatives w.r.t. the loop inputs, i.e. the input loop vars, are
// returned in `grad_outputs`.
Status AddWhileGradientLoop(WhileContext* while_ctx,
                            const std::vector<Output>& grad_inputs,
                            Output backprop_execution_pred, const Scope& scope,
                            std::vector<Output>* grad_outputs);

}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CC_FRAMEWORK_WHILE_GRADIENTS_H_
