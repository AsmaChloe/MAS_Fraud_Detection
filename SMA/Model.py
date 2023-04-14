import pandas as pd

from Agent import *

def get_nb_classifiers(ratio, nb_proposers, nb_acceptors) :
    """
        Get the number of classifiers for each type according to the ratio of each type
    :param ratio: the ratio of classifiers for each type
    :param nb_proposers: number of proposers
    :param nb_acceptors: number of acceptors
    :return: the number of classifiers for each type
    """
    N = nb_proposers + nb_acceptors
    nb_best_classifiers = int(ratio[0] * N)
    nb_average_classifiers = int(ratio[1] * N)
    nb_poor_classifiers = int(ratio[2] * N)

    if nb_best_classifiers + nb_average_classifiers + nb_poor_classifiers < N:
        nb_best_classifiers += 1

    return nb_best_classifiers, nb_average_classifiers, nb_poor_classifiers

def get_classifiers(nb_best_classifiers, nb_average_classifiers, nb_poor_classifiers) :
    """
        Get the classifiers for each type
    :param nb_best_classifiers: number of best classifiers
    :param nb_average_classifiers: number of average classifiers
    :param nb_poor_classifiers: number of poor classifiers
    :return: the classifiers for each type
    """
    classifiers = []

    for i in range(nb_best_classifiers) :
        #Pick random model from "./../modeles/best"
        model_number = randint(1, len(os.listdir("./../modeles/best")))
        model_name = os.listdir("./../modeles/best")[model_number-1]
        classifier = pickle.load(open("./../modeles/best/"+model_name, 'rb'))
        classifiers.append(classifier)

    for i in range(nb_average_classifiers) :
        #Pick random model from "./../modeles/average"
        model_number = randint(1, len(os.listdir("./../modeles/average")))
        model_name = os.listdir("./../modeles/average")[model_number-1]
        classifier = pickle.load(open("./../modeles/average/"+model_name, 'rb'))
        classifiers.append(classifier)

    for i in range(nb_poor_classifiers) :
        #Pick random model from "./../modeles/poor"
        model_number = randint(1, len(os.listdir("./../modeles/poor")))
        model_name = os.listdir("./../modeles/poor")[model_number-1]
        classifier = pickle.load(open("./../modeles/poor/"+model_name, 'rb'))
        classifiers.append(classifier)

    return classifiers

class Model(mesa.Model) :
    """A model based on the Paxos algorithm"""

    def __init__(self, nb_proposers, nb_acceptors, nb_learners, data = None, situation = None, ratio=(1,0,0), nb_reload=0) :
        assert(sum(ratio) == 1)

        if nb_reload >= 50 :
            print("Too many reloads, consider changing the ratio of classifiers or the number of agents")
            return

        #Create a scheduler
        self.schedule = mesa.time.BaseScheduler(self)
        self.data = data
        self.consensus = None
        self.nb_crashes = 0
        self.nb_reload = nb_reload

        #Get the number of classifiers for each type
        nb_best_classifiers, nb_average_classifiers, nb_poor_classifiers = get_nb_classifiers(ratio, nb_proposers, nb_acceptors)
        classifiers = get_classifiers(nb_best_classifiers, nb_average_classifiers, nb_poor_classifiers)

        if situation is None :
            #Number of proposers
            self.num_proposers = nb_proposers
            #Number of acceptors
            self.num_acceptors = nb_acceptors
            #Number of learners
            self.num_learners = nb_learners

            # Create agents
            for i in range(self.num_proposers):
                a = Proposer(i, self, classifier=classifiers.pop(randint(0, len(classifiers)-1)))
                self.schedule.add(a)
            for i in range(self.num_acceptors):
                b = Acceptor(nb_proposers + i, self, classifier=classifiers.pop(randint(0, len(classifiers)-1)))
                self.schedule.add(b)
            for i in range(self.num_learners):
                c = Learner(nb_proposers + nb_acceptors + i, self)
                self.schedule.add(c)
        elif situation == 1 :
            self.num_proposers = 2
            self.num_acceptors = 5
            self.num_learners = 1
            # Create agents
            for i in range(self.num_proposers):
                a = Proposer(i, self, i)
                self.schedule.add(a)
            for i in range(self.num_acceptors):
                b = Acceptor(nb_proposers + i, self, int(i%2 == 0) )
                self.schedule.add(b)
            for i in range(self.num_learners):
                c = Learner(nb_proposers + nb_acceptors + i, self)
                self.schedule.add(c)
        elif situation == 2 :
            self.num_proposers = 2
            self.num_acceptors = 4
            self.num_learners = 1
            # Create agents
            for i in range(self.num_proposers):
                a = Proposer(i, self, i)
                self.schedule.add(a)
            for i in range(self.num_acceptors):
                b = Acceptor(nb_proposers + i, self, int(i%2 == 0) )
                self.schedule.add(b)
            for i in range(self.num_learners):
                c = Learner(nb_proposers + nb_acceptors + i, self)
                self.schedule.add(c)
        elif situation == 3 :
            self.num_proposers = 2
            self.num_acceptors = 4
            self.num_learners = 1
            # Create agents
            for i in range(self.num_proposers):
                a = Proposer(i, self, 0)
                self.schedule.add(a)
            for i in range(self.num_acceptors):
                b = Acceptor(nb_proposers + i, self, int(i%2 == 0) )
                self.schedule.add(b)
            for i in range(self.num_learners):
                c = Learner(nb_proposers + nb_acceptors + i, self)
                self.schedule.add(c)
        elif situation == 4 :
            self.num_proposers = 2
            self.num_acceptors = 4
            self.num_learners = 1
            # Create agents
            for i in range(self.num_proposers):
                a = Proposer(i, self, 0)
                self.schedule.add(a)
            for i in range(self.num_acceptors):
                b = Acceptor(nb_proposers + i, self, int(bool(i)) )
                self.schedule.add(b)
            for i in range(self.num_learners):
                c = Learner(nb_proposers + nb_acceptors + i, self)
                self.schedule.add(c)

        #Model state
        self.finished = False

        if not self.check_consensus_possible() :
            print("No consensus possible... Reloading...")
            self.__init__(nb_proposers, nb_acceptors, nb_learners, data, situation, ratio, nb_reload=nb_reload+1)

        self.majority = self.num_acceptors // 2 + 1
        #
        # for agent in self.schedule.agents :
        #     if isinstance(agent, Proposer) or isinstance(agent, Acceptor) :
        #         print(agent.classifier, agent.predicted_value)

    def step(self):
        all_finished = True
        for agent in self.schedule.agents :
            if agent.state != "Finished" :
                all_finished = False
                break
        if all_finished :
            self.finished = True
            self.consensus = self.schedule.agents[0].consensus[1]
        else :
            self.schedule.step()

    def check_consensus_possible(self):
        proposers = [agent for agent in self.schedule.agents if isinstance(agent, Proposer)]
        acceptors = [agent for agent in self.schedule.agents if isinstance(agent, Acceptor)]

        #Check repartition of values
        #Predicted values by acceptors
        acceptor_predicted_values = {0:0, 1:0}
        for acceptor in acceptors :
            if acceptor.predicted_value in acceptor_predicted_values :
                acceptor_predicted_values[acceptor.predicted_value] += 1

        if(acceptor_predicted_values[0]==acceptor_predicted_values[1]):
            print("50/50")
            #No consensus possible
            return False
        else :

            proposer_predicted_values = {0:0, 1:0}
            for proposer in proposers :
                if proposer.predicted_value in proposer_predicted_values :
                    proposer_predicted_values[proposer.predicted_value] += 1
                if proposer_predicted_values[0]==proposer_predicted_values[1]:
                    return True

            #Get the most predicted value by acceptors
            nb_most_predicted_acceptor = 0
            most_predicted_value_acceptor = None
            for value in acceptor_predicted_values :
                if acceptor_predicted_values[value] > nb_most_predicted_acceptor :
                    nb_most_predicted_acceptor = acceptor_predicted_values[value]
                    most_predicted_value_acceptor = value

            #Get the most predicted value by proposers
            nb_most_predicted_proposer = 0
            most_predicted_value_proposer = None
            for value in proposer_predicted_values :
                if proposer_predicted_values[value] > nb_most_predicted_proposer :
                    nb_most_predicted_proposer = proposer_predicted_values[value]
                    most_predicted_value_proposer = value
            # print(proposer_predicted_values, acceptor_predicted_values)
            if most_predicted_value_acceptor == most_predicted_value_proposer :
                #Consensus possible
                return True
            else :
                return False


    #     proposers = [agent for agent in self.schedule.agents if isinstance(agent, Proposer)]
    #     acceptors = [agent for agent in self.schedule.agents if isinstance(agent, Acceptor)]
    #
    #     #Check repartition of values
    #
    #     #Predicted values by acceptors
    #     predicted_values = {}
    #     for acceptor in acceptors :
    #         if acceptor.predicted_value in predicted_values :
    #             predicted_values[acceptor.predicted_value] += 1
    #         else :
    #             predicted_values[acceptor.predicted_value] = 1
    #
    #     #Get the most predicted value by acceptors
    #     nb_most_predicted = 0
    #     most_predicted_value = None
    #     for value in predicted_values :
    #         if predicted_values[value] > nb_most_predicted :
    #             nb_most_predicted = predicted_values[value]
    #             most_predicted_value = value
    #
    #
    #     if nb_most_predicted < self.num_acceptors // 2 + 1 :
    #         #No consensus possible according to acceptors
    #         #Check if there is a majority of proposers with the same value
    #
    #         for proposer in proposers :
    #             if proposer.predicted_value in predicted_values :
    #                 predicted_values[proposer.predicted_value] += 1
    #             else :
    #                 predicted_values[proposer.predicted_value] = 1
    #
    #         nb_most_predicted = 0
    #         most_predicted_value = None
    #         for value in predicted_values :
    #             if predicted_values[value] > nb_most_predicted :
    #                 nb_most_predicted = predicted_values[value]
    #                 most_predicted_value = value
    #
    #         if nb_most_predicted >= (self.num_acceptors + self.num_proposers) // 2 + 1 : #TODO
    #             #If there is a majority once the proposers are taken into account, consensus is possible
    #             #To do so, we add one acceptor with the most predicted value
    #             self.num_acceptors += 1
    #             self.schedule.add(Acceptor(self.schedule.agents[-1].unique_id+1, self, most_predicted_value))
    #         else :
    #             #No consensus possible ; reset the model
    #             self.__init__(self.num_proposers, self.num_acceptors, self.num_learners)
    #     else :
    #         #Check if there is at least one proposer with the most predicted value
    #         found = False
    #         for proposer in proposers :
    #             if proposer.predicted_value == most_predicted_value :
    #                 found = True
    #                 break
    #         if not found :
    #             self.num_proposers += 1
    #             self.schedule.add(Proposer(self.schedule.agents[-1].unique_id+1, self, most_predicted_value))
