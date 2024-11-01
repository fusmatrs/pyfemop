
import numpy as np
from pymoo.core.problem import Problem
from pymoo.problems.static import StaticProblem
from pymoo.core.evaluator import Evaluator
from pymoo.core.algorithm import Algorithm
from pymoo.core.termination import Termination
import dill
from pathlib import Path

from mooseherder.mooseherd import MooseHerd
from mooseherder import SweepReader
from mooseherder import ExodusReader

import pickle
import copy
import pyvista as pv
from pyfemop.optimisationmanager.costfunctions import CostFunction
from pycoatl.spatialdata.importsimdata import simdata_to_spatialdata

import multiprocessing as mp

class OptimisationInputs():

    def __init__(self,parameter_space : dict ,algorithm : Algorithm ,termination : Termination, run_type = 'default', base_params : dict = None):
        """Class for containing the input choices for an optimisation.

        Args:
            parameter_space (dict): The parameters to optimise on, keys must match the names in the files.
            algorithm (Algorithm): Pymoo algorithm to use for the optimisation
            termination (Termination): Pymoo termination criteria to use for the optimisation
            run_type (str) : Default 'default'. Type of run: 'default', 'sensitivity' or 'gradient'.
            base_params (dict) : Default None. Dictionary of parameters used as a baseline for the sensitivity runs
        """
        self._algorithm = algorithm
        self._parameter_space = parameter_space
        self._opt_parameters = [x for x in parameter_space.keys()]
        self._n_var = len(parameter_space)
        lb=[]
        ub = []
        for value in parameter_space.values():
            lb.append(value[0])
            ub.append(value[1])

        self._bounds = (np.array(lb),np.array(ub))
        self._termination = termination
        self._run_type = run_type
        self._base_params = base_params
        # Check base params exist for sensitivity run
        if self._run_type == 'sensitivity' and self._base_params is None:
            raise ValueError('Sensitivity optimisation runs require a set of base parameters.')


class MooseOptimisationRun():

    def __init__(self,name : str,optimisation_inputs : OptimisationInputs,herd : MooseHerd,cost_function : CostFunction,data_filter = None):
        """Class to contain everything needed for an optimization run 
        with moose. Should be pickle-able.

        Args:
            name (str) : string to name the run.
            herd (MooseHerd): MooseHerd instance for the run.
            costfunction (CostFunction): CostFunction instance should operate on data or sensitivities as required.
            data_filter : Instance of data filter class or None.
        """
        self._name = name
        self._herd = herd
        self._optimisation_inputs = optimisation_inputs
        self._cost_function = cost_function
        self._data_filter = data_filter
        self._algorithm = optimisation_inputs._algorithm
        self.assign_parameter_list()
 
        # Set up problem
        self._problem = Problem(n_var=optimisation_inputs._n_var,
                  n_obj=self._cost_function._n_obj,
                  xl=optimisation_inputs._bounds[0],
                  xu=optimisation_inputs._bounds[1])
               
        # Setup algorithm
        self._algorithm.setup(self._problem,termination=optimisation_inputs._termination)
        
        # Get output reader
        self.sweep_reader = SweepReader(herd._dir_manager,num_para_read=4)
        
    def assign_parameter_list(self):
        """Iterate through the herder modifiers and work out where each parameter goes.
        """
        parameter_assignment = []
        for modifier in self._herd._modifiers:
            temp_params = [x for x in modifier._vars.keys()]
            parameter_assignment.append([x for x in temp_params if x in self._optimisation_inputs._opt_parameters])
        
        self._parameter_assignment = parameter_assignment

    def run(self,num_its):
        """Run the optimization for n_its number of generations.
        Only if the algorithm hasn't terminated.
        Does different things for different run types
        Current types:
        -Default

        Args:
            num_its (int): _description_
        """
        print('************************************************')
        print('             <<   Run Start   >>                ')
        print('------------------------------------------------')
        print('              All Input Parameters              ')

        for i,mod in enumerate(self._herd._modifiers):
            print('Modifier {}'.format(i+1))
            print('-----------')
            for key in mod._vars:
                print('{} = {}'.format(key,mod._vars[key]))
        for n_gen in range(num_its):
            #Check if termination criteria has been met. 
            if not self._algorithm.has_next():
                # Kill the loop if the algorithm has terminated.
                break
            print('************************************************')
            cur_gen = self._algorithm.n_gen
            if cur_gen is None:
                cur_gen = 1
            print('       Running Optimization Generation {}     '.format(cur_gen))
            print('------------------------------------------------')
            # Ask for the next solution to be implemented
            #Get parameters
            pop = self._algorithm.ask()
            x = pop.get("X")
            
            if self._optimisation_inputs._run_type == 'default':

                self._herd._dir_manager.clear_dirs()
                self._herd._dir_manager.create_dirs()

                #Run moose for all x.
                #Moose herder needs list of dicts. With correctly named parameters. 
                # The order of parameters will be the same as in the bounds
                
                # Convert x to list of dict
                # Need to check from the herder what is running what. i.e. what should the format be 
                para_vars = []
                for i in range(x.shape[0]):
                    sub_vars = []
                    p_no = 0
                    for param_list in self._parameter_assignment:
                        if param_list:
                            para_dict = dict()
                            for j,key in enumerate(param_list):
                                para_dict[key] = x[i,p_no]
                                p_no+=1
                        else:
                            para_dict = None
                        sub_vars.append(para_dict)
                    para_vars.append(sub_vars)
                

                self._herd.run_para(para_vars)
                print('        Run time = {:.2f} seconds.'.format(self._herd.get_sweep_time()))
                print('------------------------------------------------')
                # Read in moose results and get cost. 
                print('                Reading Data                    ')
                print('------------------------------------------------')
                
                #output_files = self.sweep_reader.read_all_output_keys()
                data_list = self.sweep_reader.read_results_sequential()
                # Convert data list to SpatialData
                spatial_data_list = []
                for data in data_list:
                  spatial_data_list.append(simdata_to_spatialdata(data[-1]))
                
                # Run Data filter, if there is one.
                if self._data_filter is not None:
                    print('             Running Data Filter                ')
                    print('------------------------------------------------')
                    # Old code, runs sequential
                    #filtered_data_list = []
                    #for spatial_data in spatial_data_list:
                    #    filtered_data_list.append(self._data_filter.run_filter(spatial_data))
                    #spatial_data_list = filtered_data_list
                    # New  runs parallel
                    spatial_data_list = self._data_filter.run_filter(spatial_data_list)
                print('            Calculating Objectives              ')
                print('------------------------------------------------')

                costs = np.array(self._cost_function.evaluate_parallel(spatial_data_list))
                
                F = []
                for i in range(costs.shape[1]):
                    F.append(costs[:,i])
            
            elif self._optimisation_inputs._run_type == 'sensitivity':
                # Sensitivity based run
                all_costs = np.empty(x.shape[0]) 
                sweep_params = list([])
                for i in range(x.shape[0]):
                    self._herd._dir_manager.clear_dirs()
                    self._herd._dir_manager.create_dirs()
                    #Check params are valid
                    #if x[i][0] + x[i][1] < 1:
                    #    all_costs[i]=1000
                    #    continue
                    print('Populate sweep Parameters')
                    
                    
                    # Assumes there's a neml2 input for now...
                    # Could be the case that there's 2 modifiers, one for gmsh and one for moose
                    # Easy fix would just be to check # of input modifiers in herd, but seems lazy
                    para_vars = []
                    for i in range(x.shape[0]):
                        sub_vars = []
                        p_no = 0
                        for param_list in self._parameter_assignment:
                            if param_list:
                                para_dict = dict()
                                for j,key in enumerate(param_list):
                                    para_dict[key] = x[i,p_no]
                                    p_no+=1
                            else:
                                para_dict = None
                            sub_vars.append(para_dict)
                        para_vars.append(sub_vars)
                    #print(para_vars)

                    sweep_params = []
                    for l in para_vars:
                        r = l.copy()
                        r[-1] = self._optimisation_inputs._base_params.copy()
                        sweep_params.append(r)
                        for p in self._optimisation_inputs._base_params:
                            new_dict = self._optimisation_inputs._base_params.copy()
                            new_dict[p] = self._optimisation_inputs._base_params[p]*1.1
                            r = l.copy()
                            r[-1] = new_dict
                            sweep_params.append(r)
                    
                print(sweep_params)
                #print('Run the herd')
                self._herd.run_para(sweep_params)
                print('        Run time = {:.2f} seconds.'.format(self._herd.get_sweep_time()))
                print('------------------------------------------------')
                # Read in moose results and get cost. 
                print('                Reading Data                    ')
                print('------------------------------------------------')
                data_list = self.sweep_reader.read_results_sequential()

                spatial_data_list = []
                for data in data_list:
                    spatial_data_list.append(simdata_to_spatialdata(data[-1]))
                
                #print('Calculate costs')
                # First step is to batch runs into individual sensitivity runs. 
                # The calculate the sensitivity of each batch
                nr= len(self._optimisation_inputs._base_params)
                sens = []
                run_fails = np.full(x.shape[0],False)

                # Assemble the sensitivity runs into batches. (i.e. all runs on same geometry)
                batch_list = []
                for i in range(x.shape[0]):
                    batch_list.append(spatial_data_list[i*(nr+1):i*(nr+1)+(nr+1)])
                
                strain_field = 'mechanical_strain'
                # Here should go the data filter.
                if self._data_filter is not None:
                    print('             Running Data Filter                ')
                    print('------------------------------------------------')
                    strain_field = 'filtered_strain'
                    n_threads = mp.cpu_count() - 1#len(spatial_data_list)

                    with mp.Pool(n_threads) as pool:
                        processes = []
                        for batch in batch_list:
                            processes.append(pool.apply_async(self._data_filter.run_filter, (batch,))) # tuple is important, otherwise it unpacks strings for some reason
                        f_list=[pp.get() for pp in processes]
                    batch_list = f_list

                    #print(batch)
                for i,batch in enumerate(batch_list):
                    sens_temp = []
                    base_file = batch[0]#simdata_to_spatialdata(spatial_data_list[-1])
                    base_file.get_equivalent_strain(strain_field)
                    if base_file.time[-1] != self._cost_function._endtime:
                        run_fails[i] = True
                    for alt_file in batch[1:]:
                        if alt_file.time[-1] != self._cost_function._endtime:
                            run_fails[i] = True
                        alt_file.get_equivalent_strain(strain_field)
                        sens_temp.append(base_file.data_fields['equiv_strain'].data[:,0,-1]-alt_file.data_fields['equiv_strain'].data[:,0,-1])
                    
                    #sens.append(np.mean(np.abs(np.array(sens_temp)),axis=1))
                    # Using RMS to make everything positive.
                    sens.append(np.sqrt(np.ravel(np.nanmean(np.array(sens_temp)**2,axis=1))))
                    
                costs = []
                for count,tsens in enumerate(sens):
                    
                    # Check if run has failed
                    if run_fails[count]:
                        costs.append(1E6)
                        continue

                    costs.append(-np.sqrt(np.sum(np.power(tsens,2))))
            
                #costs = self._cost_function.evaluate_sequential(spatial_data_list)
                

                # Calculate actual cost (based on similarity of mean sensitivities)
                #costs = []
                #for count,tsens in enumerate(sens):
                    
                    # Check if run has failed
                #    if run_fails[count]:
                #        costs.append(1E6)
                #        continue
                #    sqsum = 0
                #    for i,val in enumerate(tsens):
                #        
                #        for negval in tsens[1+i:]:
                #            sqsum+=np.power(val-negval,2)
                #    if sqsum == 0: # Check that all sensitivities aren't zero
                #        costs.append(1E6)
                #    else:
                #        costs.append(sqsum/np.sum(np.power(tsens,2)))
                
                F=costs  

            # Give the problem the updated costs. 
            static = StaticProblem(self._problem,F=F)
            self._algorithm.evaluator.eval(static,pop)
            self._algorithm.tell(infills=pop)
            self.backup()
            print('              Generation Complete               ')
            print('************************************************')
            print('')
            self.print_status_to_file()
        self.backup()
        self.print_status()
        self.print_status_to_file()

    
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

    def print_status(self):
        """Prints the current status of the optimization. 
        Designed to be human readable.
        """
        F = self._algorithm.result().F 
        X = self._algorithm.result().X

        print(self.banner())
         
        print('************************************************')
        print('               Current Status                   ')
        print('************************************************')
        print('Completed Generations: {}'.format(self._algorithm.n_gen-1))
        # Not sure why the below code doesn't work, (Returns 0) but can get n_evals roughly
        print('Completed Evaluations: {}'.format(self._algorithm.evaluator.n_eval))
        #print('Completed Evaluations: {}'.format((self._algorithm.n_gen-1)*self._algorithm.pop_size))
        # Doesn't seem like there's a way to get which termination tripped on the algorithm
        if self._algorithm.has_next():
            print('Termination criteria not reached.')
        else:
            print('Algorithm terminated.')
        print('------------------------------------------------')
        if len(X.shape)==1:
            print('      Single Objective Optimisation Result      ')
            print('------------------------------------------------')
            outstring = 'Parameters:\n'
            for j,key in enumerate(self._optimisation_inputs._opt_parameters):
                outstring += '{} = {};\n'.format(key,X[j])
            
            outstring+= '\ngives result:\n'
            for res in F:
                outstring+=' {},'.format(res)
            outstring = outstring[:-1]
            print(outstring)
            print('------------------------------------------------')
        else:
            print('    Multiobjective Optimisation Pareto Front    ')
            print('------------------------------------------------')
            for i in range(X.shape[0]):
                outstring = 'Parameters: '
                for j,key in enumerate(self._optimisation_inputs._opt_parameters):
                    outstring += '{} = {};'.format(key,X[i,j])
                
                outstring+= 'gives results:'
                for res in F:
                    outstring+=' {},'.format(res)
                outstring = outstring[:-1]
                print(outstring)
                print('------------------------------------------------')
    

    def get_backup_path(self):
        """Get a path to save the dill backup to.

        Returns:
            str: Path to the backup dill.
        """
        backup_path = self._herd._dir_manager._base_dir / (self._name.replace(' ','_').replace('.','_') + '.pickle')
        return backup_path

    def backup(self):
        """Create a pickle dump of the class instance.
        """
        #pickle_path = self._herd._base_dir + self._name.replace(' ','_').replace('.','_') + '.pickle'
        #print(pickle_path)
        with open(self.get_backup_path(),'wb') as f:
            dill.dump(self,f,dill.HIGHEST_PROTOCOL)

    @classmethod
    def restore_backup(cls,backup_path):
        """        
        Restores a run from a backup.
                      

        Parameters
        ----------
        cls : MooseOptimisationRun() instance
            Instance to be restored.
        backup_path : string
            Path to pickled file.

        Returns
        -------
        cls : MooseOptimisationRun() instance.
           Restored MooseOptimisationRun instance

        """
        
        with open(backup_path, 'rb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            cls = dill.load(f)
        
        return cls


    
    def print_status_to_file(self):
        """Prints the current status of the optimization to a file. 
        Designed to be human readable.
        """
        F = self._algorithm.result().F 
        X = self._algorithm.result().X

        outpath = self._herd._dir_manager._base_dir / (self._name.replace(' ','_').replace('.','_') + '.txt')
        with open(outpath,'a') as f:

            f.write(self.banner())
            f.write('\n')
            f.write('************************************************\n')
            f.write('               Current Status                   \n')
            f.write('************************************************\n')
            f.write('Completed Generations: {}\n'.format(self._algorithm.n_gen-1))
            # Not sure why the below code doesn't work, (Returns 0) but can get n_evals roughly
            f.write('Completed Evaluations: {}\n'.format(self._algorithm.evaluator.n_eval))
            #f.write('Completed Evaluations: {}\n'.format((self._algorithm.n_gen-1)*self._algorithm.pop_size))
            # Doesn't seem like there's a way to get which termination tripped on the algorithm
            if self._algorithm.has_next():
                f.write('Termination criteria not reached.\n')
            else:
                f.write('Algorithm terminated.\n')
            f.write('------------------------------------------------\n')
            if len(X.shape)==1:
                f.write('      Single Objective Optimisation Result      \n')
                f.write('------------------------------------------------\n')
                outstring = 'Parameters:\n'
                for j,key in enumerate(self._optimisation_inputs._opt_parameters):
                    outstring += '{} = {};\n'.format(key,X[j])
                
                outstring+= '\ngives result:\n'
                for res in F:
                    outstring+=' {};'.format(res)
                outstring = outstring[:-1]
                f.write(outstring+'\n')
                f.write('------------------------------------------------\n')
            else:
                f.write('    Multiobjective Optimisation Pareto Front    \n')
                f.write('------------------------------------------------\n')
                for i in range(X.shape[0]):
                    outstring = 'Parameters: '
                    for j,key in enumerate(self._optimisation_inputs._opt_parameters):
                        outstring += '{} = {}; '.format(key,X[i,j])
                    
                    outstring+= 'gives results:'
                    for res in F:
                        outstring+=' {};'.format(res)
                    outstring = outstring[:-1]
                    f.write(outstring+'\n')
                    #f.write('\n')
                    f.write('------------------------------------------------\n')

    def print_status_dev(self,to_file = True):
        """Prints the current status of the optimization to a file. 
        Designed to be human readable.
        """
        F = self._algorithm.result().F 
        X = self._algorithm.result().X

        #Construct enormous string.
        mega_string = ''
        mega_string += self.banner() + '\n'
        mega_string +='************************************************\n'
        mega_string +='               Current Status                   \n'
        mega_string +='************************************************\n'
        mega_string +='Completed Generations: {}\n'.format(self._algorithm.n_gen-1)
        mega_string +='Completed Evaluations: {}\n'.format((self._algorithm.n_gen-1)*self._algorithm.pop_size)
        # Doesn't seem like there's a way to get which termination tripped on the algorithm
        if self._algorithm.has_next():
            mega_string +='Termination criteria not reached.\n'
        else:
            mega_string +='Algorithm terminated.\n'
        mega_string +='------------------------------------------------\n'
        if len(X.shape)==1:
            mega_string +='      Single Objective Optimisation Result      \n'
            mega_string +='------------------------------------------------\n'
            outstring = 'Parameters:\n'
            for j,key in enumerate(self._optimisation_inputs._opt_parameters):
                outstring += '{} = {},\n'.format(key,X[j])
            
            outstring+= '\ngives result:\n'
            for res in F:
                outstring+=' {},'.format(res)
            outstring = outstring[:-1]
            mega_string += outstring+'\n'
            mega_string +='------------------------------------------------\n'
        else:
            mega_string +='    Multiobjective Optimisation Pareto Front    \n'
            mega_string +='------------------------------------------------\n'
            for i in range(X.shape[0]):
                outstring = 'Parameters: '
                for j,key in enumerate(self._optimisation_inputs._opt_parameters):
                    outstring += '{} = {}, '.format(key,X[i,j])
                
                outstring+= 'gives results:'
                for res in F:
                    outstring+=' {},'.format(res)
                outstring = outstring[:-1]
                mega_string +=outstring+'\n'
                mega_string +='------------------------------------------------\n'
        print(mega_string)
        if to_file:
            outpath = self._herd._dir_manager._base_dir / (self._name.replace(' ','_').replace('.','_') + '.txt')
            with open(outpath,'w') as f:
                f.write(mega_string)

    def banner(self):
        """ Just makes a nicely formatted banner
        """
        
        outstring =  '________________________________________________\n'
        outstring += ' _____       ______ ______ __  __  ____  _____  \n'
        outstring += '|  __ \     |  ____|  ____|  \/  |/ __ \|  __ \ \n'
        outstring += '| |__) |   _| |__  | |__  | \  / | |  | | |__) |\n'
        outstring += '|  ___/ | | |  __| |  __| | |\/| | |  | |  ___/ \n'
        outstring += '| |   | |_| | |    | |____| |  | | |__| | |     \n'
        outstring += '|_|    \__, |_|    |______|_|  |_|\____/|_|     \n'
        outstring += '        __/ |                                   \n'
        outstring += '       |___/                                    \n'
        outstring += '________________________________________________'
        return outstring
    






