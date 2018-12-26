import Data_Utils as utils
import numpy as np
import random

# 24 new children every generation
# produced from top 12 parents
#
# Mutation: 5% chance to flip an individual bit for each feature mask inside a feature mask vector

POPULATION_SIZE = 25    # defines the number of individuals in the initial population, which is to be
                        # maintained at the end of each new generation via purgePopulation()

'''
    Class: EDA
    @summary: A class providing all the necessary functionality for an Estimation of Distribution Algorithm
    @attribute:   fms     ->  The randomly generated population to be used.
    @function:    getPopAccuraciesVanderbilt()
    @function:    getPopAccuraciesCasis()
    @function:    sortByFitness()
    @function:    selectParents()
    @function:    selectParentsOptimized()
    @function:    procreate()
    @function:    mutate()
    @function:    runEvolution()
    @function:    runOptimizedEvolution()
'''
class EDA:

    def __init__(self):
        # Generate the initial population
        self.fms = utils.getFMPopulation(POPULATION_SIZE)
        # Declare a list that will contain the accuracies for each individual using a specified data set
        self.fms_accList = []


    '''
        Function: getPopAccuraciesVanderbilt()
        @summary:
        @param:   population
        @param:   n_splits
        @param:   updatedData
        @return:  accList
    '''
    def getPopAccuraciesVanderbilt(self, population, n_splits, updatedData = False):
        accList = []
        if updatedData:
            utils.ResetVanderbiltCUFile(n_splits)
        for individual in population:          # was self.fms
            accuracy = utils.evaluateLSVMAccuracy('vanderbilt', n_splits, individual)
            accList.append(round(accuracy, 3))
        return accList

    '''
        Function: getPopAccuraciesCasis()
        @summary:
        @param:   population
        @return:  accList
    '''
    def getPopAccuraciesCasis(self, population):
        accList = []
        for individual in population:
            accuracy = utils.evaluateLSVMAccuracy('casis', 4, individual)
            accList.append(round(accuracy, 3))
        return accList

    '''
        Function: sortByFitness()
        @summary: Sorts the population according to the fitness of the individuals from most fit to least fit
        !!@param       population  ->  the list of individuals for which fitness will be determined and then be sorted based on their fitnesses
        !!@param        accList    ->  the list of computed fitness values / accuracies for each individual in the specified population
        @return:
    '''
    def sortByFitness(self):
        temp_tuple = zip(self.fms, self.fms_accList)
        temp_tuple = sorted(temp_tuple, key=lambda x: x[1], reverse = True)
        for i in range(0, len(self.fms)):
            self.fms[i] = temp_tuple[i][0]
            self.fms_accList[i] = temp_tuple[i][1]

    '''
        Function: selectParents()
        @summary: Selects a specified number of individuals from the population to use for a SINGLE instance of procreation
        @param:   num_parents    ->  the number of individuals to be selected from the population
        @return:  parents        ->  list of the randomly selected individuals
    '''
    def selectParentsOptimized(self, num_parents):
        parents = []   
        prev_indexes = []

        for i in range(0, num_parents):
            # (11 = 12 - 1)
            index = random.randint(0, 9)
            while index in prev_indexes:
                index = random.randint(0, 9)
            prev_indexes.append(index)
            parents.append(self.fms[index])

        return parents

    def selectParents(self, num_parents):
        parents = []   
        prev_indexes = []

        for i in range(0, num_parents):
            # (11 = 12 - 1)
            index = random.randint(0, 11)
            while index in prev_indexes:
                index = random.randint(0, 11)
            prev_indexes.append(index)
            parents.append(self.fms[index])

        return parents

    '''
        Function: procreate()
        @summary: Generates the specified number of new individuals using the specified parents
        @param:   parents     ->  the list of parents to be used in this instance of procreation
        @return:  child       ->  the new individual that has been generated
    '''
    def procreate(self, parents):
        child = []
        for i in range(0, 95):
            #sum = 0
            #for j in range(0, len(parents)):
            index = random.randint(0, len(parents) - 1)
            child.append(parents[index][i])
            #    sum += parents[j][i]
            #density = sum/len(parents)
            #val = random.uniform(0, 1)
            #if val <= density:
            #    child.append(1)
            #else:
            #    child.append(0)
        return child

    '''
        Function: mutate()
        @summary: Iterates through a child's feature mask vector and flips the bit at a given index if a randomly
                  generated number between 0 and 1 is less than or equal to the specified mutation rate
        @param:   child           ->  the child to be mutated
        @param:   mutation_rate   ->  the specified mutation rate
        @return:  child           ->  the same child that was passed, but after mutation has been applied
    '''
    def mutate(self, child, mutation_rate):
        for i in range(0, 95):
            rand = random.uniform(0, 1)
            if rand <= mutation_rate:
                if child[i] == 1:
                    child[i] = 0
                else:
                    child[i] = 1
        return child

    '''
        Function: replacePopulation() - standard replacement method
        @summary: Replaces bottom 24 individuals from previous generation with the 24 new children
        @param:   children   ->  list containing every individual in the population
        @param:   accList    ->  list containing computed accuracies/fitness values for the children
    ''' 
    def replacePopulation(self, children, accList):
        for i in range(1, POPULATION_SIZE):
            self.fms[i] = children[i - 1] 
            self.fms_accList[i] = accList[i - 1]

    '''
        Function: replacePopulation() - optimized replacement method
        @summary: Replaces bottom 15 individuals from previous generation with the 15 new children
        @param:   children   ->  list containing every individual in the population
        @param:   accList    ->  list containing computed accuracies/fitness values for the children
    ''' 
    def replacePopulationOptimized(self, children, accList):
        for i in range(10, POPULATION_SIZE):
            self.fms[i] = children[i - 10]
            self.fms_accList[i] = accList[i - 10]

    '''
        Function: runOptimizedEvolution() - optimized evolution loop
        @summary: Runs the Estimation of Distribution algorithm for n iterations
        @param:   n                 ->  the specified number of evaluations, i.e. total number of children generated
        @param:   dataset_select    ->  the specified dataset to use for accuracy/fitness metrics
        @param:   mutation_rate     ->  the specified mutation rate to apply to newly generated children
        @return:  optimal_solution  ->  the individual with the highest computed accuracy in the final population
        @return:  avg_acc           ->  the computed average accuracy for all individuals in the final population
    '''
    def runOptimizedEvolution(self, n, dataset_select, mutation_rate):
        # Analyze fitness of initial population using selected dataset
        if (dataset_select == 0):
            self.fms_accList = self.getPopAccuraciesCasis(self.fms)
        else:
            self.fms_accList = self.getPopAccuraciesVanderbilt(self.fms, 8, False)

        for i in range(0, (int)((n - 25)/15)):
            # Sort (i - 1)th population according to their fitness values (including the initial population once the loop is first entered)
            self.sortByFitness()
            children = []
            accList = []
            # Create 15 new children for ith generation
            for j in range(0, 15):
                # Randomly determine number of parents for jth child from 2 to 10 of the top 10 individuals
                num_parents = random.randint(2, 10)
                # Randomly select num_parents parents for procreation to generate one child
                parents = self.selectParentsOptimized(num_parents)
                # Generate the child using the selected parents
                child = self.procreate(parents)
                # Mutate the child
                child = self.mutate(child, mutation_rate)
                children.append(child)
            # Analyze fitness of new generation
            if (dataset_select == 0):
                accList = self.getPopAccuraciesCasis(children)
            else:
                accList = self.getPopAccuraciesVanderbilt(children, 8, False)
            # Replace the bottom 24 individuals from (i - 1)th population with the 24 children from the ith generation
            self.replacePopulationOptimized(children, accList)
        # Once all generations have been run, sort the newest generation according to their fitness values (reverse order sorting, i.e. highest fitness to lowest fitness)
        self.sortByFitness()
        # Compute average accuracy for the final population
        sum = 0
        for i in range(0, POPULATION_SIZE):
            sum += self.fms_accList[i]
        avg_acc = sum/POPULATION_SIZE
        # After sorting, the optimal solution will be at the front
        return self.fms_accList[0], avg_acc

    '''
        Function: runEvolution() - standard evolution loop
        @summary: Runs the Estimation of Distribution algorithm for n iterations
        @param:   n                 ->  the specified number of evaluations, i.e. total number of children generated
        @param:   dataset_select    ->  the specified dataset to use for accuracy/fitness metrics
        @param:   mutation_rate     ->  the specified mutation rate to apply to newly generated children
        @return:  optimal_solution  ->  the individual with the highest computed accuracy in the final population
        @return:  avg_acc           ->  the computed average accuracy for all individuals in the final population
    '''
    def runEvolution(self, n, dataset_select, mutation_rate):
        # Analyze fitness of initial population using selected dataset
        if (dataset_select == 0):
            self.fms_accList = self.getPopAccuraciesCasis(self.fms)
        else:
            self.fms_accList = self.getPopAccuraciesVanderbilt(self.fms, 8, False)

        for i in range(0, (int)((n - 25)/24)):
            # Sort (i - 1)th population according to their fitness values (including the initial population once the loop is first entered)
            self.sortByFitness()
            children = []
            accList = []
            # Create 24 new children for ith generation
            for j in range(0, 24):
                # Randomly determine number of parents for jth child from 2 to 12 of the top 12 individuals
                num_parents = random.randint(2, 12)
                # Randomly select num_parents parents for procreation to generate one child
                parents = self.selectParents(num_parents)
                # Generate the child using the selected parents
                child = self.procreate(parents)
                # Mutate the child
                child = self.mutate(child, mutation_rate)
                children.append(child)
            # Analyze fitness of new generation
            if (dataset_select == 0):
                accList = self.getPopAccuraciesCasis(children)
            else:
                accList = self.getPopAccuraciesVanderbilt(children, 8, False)
            # Replace the bottom 24 individuals from (i - 1)th population with the 24 children from the ith generation
            self.replacePopulation(children, accList)
        # Once all generations have been run, sort the newest generation according to their fitness values (reverse order sorting, i.e. highest fitness to lowest fitness)
        self.sortByFitness()
        # Compute average accuracy for the final population
        sum = 0
        for i in range(0, POPULATION_SIZE):
            sum += self.fms_accList[i]
        avg_acc = sum/POPULATION_SIZE
        # After sorting, the optimal solution will be at the front
        return self.fms_accList[0], avg_acc

'''
print("WITHOUT Optimization...")
eda = EDA()
optimal_fm_acc, avg_acc = eda.runEvolution(5000, 0, 0.05)
print("Optimal solution accuracy for Casis-25 dataset after 5000 iterations: %s" %optimal_fm_acc)
print("Average final population accuracy: %s" %avg_acc)

eda = EDA()
optimal_fm_acc, avg_acc = eda.runEvolution(5000, 1, 0.05)
print("Optimal solution accuracy for Vanderbilt dataset after 5000 iterations: %s" %optimal_fm_acc)
print("Average final population accuracy: %s" %avg_acc)
'''
print("WITH Optimization")
eda = EDA()
optimal_fm_acc, avg_acc = eda.runOptimizedEvolution(5000, 0, 0.07)
print("Optimal solution accuracy for Casis-25 dataset after 5000 iterations: %s" %optimal_fm_acc)
print("Average final population accuracy: %s" %avg_acc)

optimal_fm_acc, avg_acc = eda.runOptimizedEvolution(5000, 0, 0.07)
print("Optimal solution accuracy for Vanderbilt dataset after 5000 iterations: %s" %optimal_fm_acc)
print("Average final population accuracy: %s" %avg_acc)
