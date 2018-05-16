#include "rbm.cpp"

/*
PSEUDOCODE:
just need to find a way to get in the usr, and the movies
we can even load the data without everything else

Just need to be able to select the users and movies with their number, somehow
*/



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
  engine.run(program, 1);//niters);

  /* Output latent factor matrices in matrix-market format */
  //output_rbm_result(training);
  test_predictions(&rbm_predict);    


  /* Report execution metrics */
  if (!quiet)
    metrics_report(m);
  return 0;
}




/**
 * GraphChi programs need to subclass GraphChiProgram<vertex-type, edge-type> 
 * class. The main logic is usually in the update function.
 */
struct RBMVerticesInMemProgramPredict : public GraphChiProgram<VertexDataType, EdgeDataType> {
  /**
   * Called before an iteration is started.
   */
  void before_iteration(int iteration, graphchi_context &gcontext) {
    reset_rmse(gcontext.execthreads);
  }



  /**
   * Called after an iteration has finished.
   */
  void after_iteration(int iteration, graphchi_context &gcontext) {
    rbm_alpha *= rbm_mult_step_dec;
    training_rmse(iteration, gcontext);
    if (iteration >= 0)
      run_validation(pvalidation_engine, gcontext);
    else std::cout<<std::endl;
  }

  /**
   *  Vertex update function.
   */
  void update(graphchi_vertex<VertexDataType, EdgeDataType> &vertex, graphchi_context &gcontext) {
        
    /*
    if (gcontext.iteration == 0){
      if (is_user(vertex.id()) && vertex.num_outedges() > 0){
        vertex_data& user = latent_factors_inmem[vertex.id()];
        user.pvec = zeros(D*3);
        for(int e=0; e < vertex.num_outedges(); e++) {
          rbm_movie mov = latent_factors_inmem[vertex.edge(e)->vertex_id()];
          float observation = vertex.edge(e)->get_data();                
          int r = (int)(observation/rbm_scaling);
          assert(r < rbm_bins);
          mov.bi[r]++;
        }
      }
      return;
    }
    else if (gcontext.iteration == 1){
      if (vertex.num_inedges() > 0){
        rbm_movie mov = latent_factors_inmem[vertex.id()]; 
        setRand2(mov.w, D*rbm_bins, 0.001);
        for(int r = 0; r < rbm_bins; ++r){
          mov.bi[r] /= (double)vertex.num_inedges();
          mov.bi[r] = log(1E-9 + mov.bi[r]);
         
          if (mov.bi[r] > 1000){
            assert(false);
            logstream(LOG_FATAL)<<"Numerical overflow" <<std::endl;
          }
        }
      }

      return; //done with initialization
    }
    */
    //go over all user nodes
    if (is_user(vertex.id()) && vertex.num_outedges()){
      vertex_data & user = latent_factors_inmem[vertex.id()]; 
      /*
      user.pvec = zeros(3*D);
      rbm_user usr(user);
      vec v1 = zeros(vertex.num_outedges()); 
      //go over all ratings
      for(int e=0; e < vertex.num_outedges(); e++) {
        float observation = vertex.edge(e)->get_data();                
        rbm_movie mov = latent_factors_inmem[vertex.edge(e)->vertex_id()];
        int r = (int)(observation / rbm_scaling);
        assert(r < rbm_bins);  
        for(int k=0; k < D; k++){
          usr.h[k] += mov.w[D*r + k];
          assert(!std::isnan(usr.h[k]));
        }
      }

      for(int k=0; k < D; k++){
        usr.h[k] = sigmoid(usr.h[k]);
        if (drand48() < usr.h[k]) 
          usr.h0[k] = 1;
        else usr.h0[k] = 0;
      }


      int i = 0;
      double prediction;
      for(int e=0; e < vertex.num_outedges(); e++) {
        rbm_movie mov = latent_factors_inmem[vertex.edge(e)->vertex_id()];
        float observation = vertex.edge(e)->get_data();
        predict1(usr, mov, observation, prediction);    
        int vi = (int)(prediction / rbm_scaling);
        v1[i] = vi;
        i++;
      }

      i = 0;
      for(int e=0; e < vertex.num_outedges(); e++) {
        rbm_movie mov = latent_factors_inmem[vertex.edge(e)->vertex_id()];
        int r = (int)v1[i];
        for (int k=0; k< D;k++){
          usr.h1[k] += mov.w[r*D+k];
        }
        i++;
      }

      for (int k=0; k < D; k++){
        usr.h1[k] = sigmoid(usr.h1[k]);
        if (drand48() < usr.h1[k]) 
          usr.h1[k] = 1;
        else usr.h1[k] = 0;
      }
      */

      int i = 0;
      for(int e=0; e < vertex.num_outedges(); e++) {
        rbm_movie mov = latent_factors_inmem[vertex.edge(e)->vertex_id()];
        float observation = vertex.edge(e)->get_data();
        double prediction;
        // Put prediction into prediction pointer
        // user is vertex_data, mov is rbm_movie, observation is rating, float
        rbm_predict(user, mov, observation, prediction, NULL);
        double pui = prediction / rbm_scaling;
        double rui = observation / rbm_scaling;
        rmse_vec[omp_get_thread_num()] += (pui - rui) * (pui - rui);
        //nn += 1.0;
        /*
        int vi0 = (int)(rui);
        int vi1 = (int)v1[i];
        for (int k = 0; k < D; k++){
          mov.w[D*vi0+k] += rbm_alpha * (usr.h0[k] - rbm_beta * mov.w[vi0*D+k]);
          assert(!std::isnan(mov.w[D*vi0+k]));
          mov.w[D*vi1+k] -= rbm_alpha * (usr.h1[k] + rbm_beta * mov.w[vi1*D+k]);
          assert(!std::isnan(mov.w[D*vi1+k]));
        }
        */
        i++; 
      }
    }
  }    
};

