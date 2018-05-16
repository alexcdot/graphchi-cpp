/**
 * @file
 * @author  Danny Bickson
 * @version 1.0
 *
 * @section LICENSE
 *
 * Copyright [2012] [Carnegie Mellon University]
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.

 *
 * @section DESCRIPTION
 * Matrix factorization using RBM (Restricted Bolzman Machines) algorithm.
 * Algorithm is described in the paper:
 * G. Hinton. A Practical Guide to Training Restricted Boltzmann Machines. University of Toronto Tech report UTML TR 2010-003
 * 
 */
#include "rbm.hpp"

int main(int argc, const char ** argv) {

  print_copyright();

  //* GraphChi initialization will read the command line arguments and the configuration file. */
  graphchi_init(argc, argv);

  /* Metrics object for keeping track of performance counters
     and other information. Currently required. */
  metrics m("rbm-inmemory-factors");

  /* Basic arguments for RBM algorithm */
  rbm_bins      = get_option_int("rbm_bins", rbm_bins);
  rbm_alpha     = get_option_float("rbm_alpha", rbm_alpha);
  rbm_beta      = get_option_float("rbm_beta", rbm_beta);
  rbm_mult_step_dec  = get_option_float("rbm_mult_step_dec", rbm_mult_step_dec);
  rbm_scaling   = get_option_float("rbm_scaling", rbm_scaling);

  parse_command_line_args();
  parse_implicit_command_line();

  mytimer.start();

  /* Preprocess data if needed, or discover preprocess files */
  int nshards = convert_matrixmarket<float>(training);

  rbm_init();

  if (validation != ""){
    int vshards = convert_matrixmarket<EdgeDataType>(validation, 0, 0, 3, VALIDATION);
    init_validation_rmse_engine<VertexDataType, EdgeDataType>(pvalidation_engine, vshards, &rbm_predict);
  }

  /* load initial state from disk (optional) */
  std::cout << training << std::endl;
  if (load_factors_from_file){
    load_matrix_market_matrix(training + "_U.mm", 0, 3*D);
    load_matrix_market_matrix(training + "_V.mm", M, rbm_bins*(D+1));
   }

  print_config();

  /* Run */
  RBMVerticesInMemProgram program;
  graphchi_engine<VertexDataType, EdgeDataType> engine(training, nshards, false, m); 
  set_engine_flags(engine);
  pengine = &engine;
  engine.run(program, niters);

  /* Output latent factor matrices in matrix-market format */
  output_rbm_result(training);
  test_predictions(&rbm_predict);    


  /* Report execution metrics */
  if (!quiet)
    metrics_report(m);
  return 0;
}
