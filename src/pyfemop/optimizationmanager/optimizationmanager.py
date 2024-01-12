# %%
# Intending to use Pymoo to optimize. 
#

import numpy as np
from pymoo.core.problem import Problem
from pymoo.problems.static import StaticProblem
from pymoo.core.evaluator import Evaluator

from mooseherder.mooseherd import MooseHerd

import pickle
import copy

class MooseOptimizationRun():

    def __init__(self,name,algorithm,termination,herd,cost_function,parameter_space):
        """Class to contain everything needed for an optimization run 
        with moose. Should be pickle-able.

        Args:
            name (str) : string to name the run.
            algorithm (pymoo algorithm): Choice of algorithm for the optimization
            termination (pymoo termination): Termination criteria for the algorithm
            herd (MooseHerd): MooseHerd instance for the run.
            costfunction (CostFunction): CostFunction instance.
            parameter_space (dict): Dict with parameters as keys and 2-tuple with lower and upper bounds as values.
        """
        self._name = name
        self._algorithm = algorithm
        self._herd = herd
        self._cost_function = cost_function
        self._parameter_space = parameter_space
        self._opt_parameters = [x for x in parameter_space.keys()]
        self._n_var = len(parameter_space)
        self._n_obj = self._cost_function.n_obj
        lb=[]
        ub = []
        for value in parameter_space.values():
            lb.append(value[0])
            ub.append(value[1])

        self._bounds = (np.array(lb),np.array(ub))
        self._termination = termination
        self._reader = cost_function._reader # Data reader 

        self._problem = Problem(n_var=self._n_var,
                  n_obj=self._n_obj,
                  xl=self._bounds[0],
                  xu=self._bounds[1])
        
        self.assign_parameters()
        
        # Setup algorithm
        self._algorithm.setup(self._problem,termination=termination)

    def assign_parameters(self):
        """Get lists of parameters for moose and gmsh from the herd 
        and determine which optimised parameters belong where.
        """

        # Default assumption is we will be running moose and gmsh
        self._mod_moose = True
        self._mod_gmsh = True

        self._moose_params = [x for x in self._herd._moose_modifier._vars.keys()]
        if self._herd._gmsh_modifier is not None:
            self._gmsh_params = [x for x in self._herd._gmsh_modifier._vars.keys()]
        else:
            # If there's no gmsh runner, then we're not running gmsh
            self._gmsh_params = []
            #self._mod_gmsh = False # Assuming you'd only want to run gmsh if you were updating the geometry
        
        self._moose_opt_params = list(set(self._moose_params)&set(self._opt_parameters))
        self._gmsh_opt_params = list(set(self._gmsh_params)&set(self._opt_parameters))
        
        if not self._moose_opt_params:
            self._mod_moose = False
        # Catch the case where we may want to run gmsh but not update it.
        if not self._gmsh_opt_params:
            self._mod_gmsh = False

    def backup(self):
        """Create a pickle dump of the class instance.
        """
        pickle_path = self._herd._base_dir + self._name.replace(' ','_').replace('.','_') + '.pickle'
        #print(pickle_path)
        with open(pickle_path,'wb') as f:
            pickle.dump(self,f,pickle.HIGHEST_PROTOCOL)


    def run(self,num_its):
        """Run the optimization for n_its number of generations.
        Only if the algorithm hasn't terminated.

        Args:
            num_its (int): _description_
        """
        for n_gen in range(num_its):
            #Check if termination criteria has been met. 
            if not self._algorithm.has_next():
                # Kill the loop if the algorithm has terminated.
                break
            print('*****Running Optimization Generation {}*****'.format(self._algorithm.n_gen))
            # Ask for the next solution to be implemented
            self._herd.clear_dirs()
            self._herd.create_dirs()
            pop = self._algorithm.ask()
            
            #Get parameters
            x = pop.get("X")

            #Run moose for all x.
            #Moose herder needs list of dicts. With correctly named parameters. 
            # The order of parameters will be the same as in the bounds
            
            # Convert x to list of dict
            para_vars = []
            for i in range(x.shape[0]):
                para_dict = dict()
                for j,key in enumerate(self._opt_parameters):
                    para_dict[key] = x[i,j]
                para_vars.append(para_dict)
            
            if self._mod_moose == False: # Don't need to change moose
                moose_vars = [self._herd._moose_modifier.get_vars()] 
            else: 
                moose_vars = [{key: l[key] for key in self._moose_opt_params} for l in para_vars]

            if self._mod_gmsh == False: # Don't need to change gmsh
                gmsh_vars = None 
            else: 
                gmsh_vars = [{key: l[key] for key in self._gmsh_opt_params} for l in para_vars]
   
            self._herd.run_para(moose_vars,gmsh_vars)
            print('Run time = '+str(self._herd.get_sweep_time())+' seconds')

            # Read in moose results and get cost. 
            print('*****Reading Data*****')
            if self._reader is not None:
                data_list = self._herd.read_results_para_generic(self._reader)
            else:
                # For working with examples relying on herder only.
                vars_to_read = ['disp_y']
                data_list = self._herd.read_results_para(vars_to_read,self._herd._sweep_iter)

            costs = np.array(self._cost_function.evaluate_parallel(data_list))
            F = []
            for i in range(costs.shape[1]):
                F.append(costs[:,i])
            

            static = StaticProblem(self._problem,F=F)
            Evaluator().eval(static,pop)

            self._algorithm.tell(infills=pop)
            self.backup()
            print('**** Generation Complete ****')

    
    def run_optimal(self,pf_nums):
        """Run a model from the pareto front

        Args:
            pf_num (list of int): Integer of the pareto front optimum to run.
        """
        
        f = self._algorithm.result().F[pf_nums] 
        x = self._algorithm.result().X[pf_nums]
        
        print('The selected parameters are: {}'.format(x))
        print('The pareto front position is: {}'.format(f))

        # Create runner
        # Temp herder
        temp_herd = copy.deepcopy(self._herd)
        temp_herd.clear_dirs()
        # There's some kind of bug with create_dirs, doesn' create via para_opts
        temp_herd.set_names(sub_dir='moose-opt')
        temp_herd.para_opts(n_moose=len(pf_nums),tasks_per_moose=1,threads_per_moose=1,redirect_out=False)
        temp_herd.create_dirs()
        
        moose_vars = [self._herd._moose_modifier.get_vars()] 

        para_vars = list()
        para_names = [t for t in self._parameter_space.keys()]
        for i in range(x.shape[0]):
            para_dict = dict()
            for j in range(len(para_names)):
                para_dict[para_names[j]] = x[i,j]
            para_vars.append(para_dict)
        
        print('**** Running Selected Models ****')
        temp_herd.run_para(moose_vars,para_vars)  


    def run_test(self,num_its):
        """Run the optimization for n_its number of generations.
        Only if the algorithm hasn't terminated.

        Args:
            num_its (int): _description_
        """
        for n_gen in range(num_its):
            #Check if termination criteria has been met. 
            if not self._algorithm.has_next():
                # Kill the loop if the algorithm has terminated.
                break
            print('*****Running Optimization Generation {}*****'.format(self._algorithm.n_gen))
            # Ask for the next solution to be implemented
            self._herd.clear_dirs()
            self._herd.create_dirs()
            pop = self._algorithm.ask()
            
            #Get parameters
            x = pop.get("X")

            #Run moose for all x.
            #Moose herder needs list of dicts. With correctly named parameters. 
            # The order of parameters will be the same as in the bounds
            
            # Convert x to list of dict
            para_vars = []
            for i in range(x.shape[0]):
                para_dict = dict()
                for j,key in enumerate(self._opt_parameters):
                    para_dict[key] = x[i,j]
                para_vars.append(para_dict)
            
            if self._mod_moose == False: # Don't need to change moose
                moose_vars = [self._herd._moose_modifier.get_vars()] 
            else: 
                moose_vars = [{key: l[key] for key in self._moose_opt_params} for l in para_vars]

            if self._mod_gmsh == False: # Don't need to change gmsh
                gmsh_vars = None 
            else: 
                gmsh_vars = [{key: l[key] for key in self._gmsh_opt_params} for l in para_vars]
            
            print(moose_vars)
            print(gmsh_vars)