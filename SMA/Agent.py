import os
import pickle
from random import randint, random
import mesa

#Last ID is a global variable
last_id = 0

#Probability of breaking down
p_breakdown = 0.2

class Paxos_Agent(mesa.Agent):
    """
    An agent as defined in the Paxos algorithm ; it can be a proposer, an acceptor or a learner
    """
    def __init__(self, unique_id, model, state, consensus):
        """
        Create a new agent
        :param unique_id: id of the agent
        :param model: the model the agent belongs to
        :param state: the state of the agent
        :param consensus: contains the consensus if the agent has reached it (state == Finished); otherwise contains proposals of concensus
        """
        super().__init__(unique_id, model)

        self.state = state
        self.consensus = consensus

    def receive_consensus(self, proposal : tuple) :
        """
            Receive consensus from proposer
            :param proposal: the couple agreed upon
            :return: None
        """
        self.state = "Finished"
        self.consensus = proposal

class Paxos_Predictive_Agent(Paxos_Agent):
    def __init__(self, unique_id, model, state, consensus, predicted_value, classifier):
        super().__init__(unique_id, model, state, consensus)

        # #choose random number between 1 and the number of files in the folder "modeles"
        # model_number = randint(1, len(os.listdir("./../modeles")))
        # #get the name of the file at the index model_number
        # model_name = os.listdir("./../modeles")[model_number-1]
        self.classifier = classifier

        if predicted_value is None:
            self.predicted_value = self.classifier.predict(self.model.data)[0]
        else:
            self.predicted_value = predicted_value

class Proposer(Paxos_Predictive_Agent):
    """
        An agent that propose values to acceptors
    """

    def __init__(self, unique_id, model, predicted_value = None, classifier = None):
        super().__init__(unique_id, model, "Proposing", (None, None), predicted_value, classifier)

        # List of responses from acceptors
        self.responses = None
        self.proposed_couple = None


    def step(self):
        # Check if the agent is broken

        # Proposing value to acceptors
        if self.state == "Proposing":

            #Send prepare request to all acceptors
            responses = self.send_prepare()
            self.responses = responses

            self.state = "Waiting Prepare"

        # Waiting for acceptors to respond to PROPOSE message
        elif self.state == "Waiting Prepare":
            #Get total of responses that are True (first element of the tuple)
            total_responses = sum(self.responses)
            #If majority of acceptors have responded True, then change state to "Accepted"
            if total_responses >= self.model.majority :
                self.state = "Accepted"
            else :
                self.state = "Rejected"
            # print("Proposer", self.unique_id, "(",self.state,") : PREPARE", self.proposed_couple, "has been", self.state)

        # Prepare request has been accepted by acceptors ; sending accept request
        elif self.state == "Accepted":
            responses = self.send_accept_request()
            self.responses = responses
            self.state = "Waiting Accept"

        # If prepare request has been rejected by acceptors, then change TIMEOUT and change state to "Proposing"
        elif self.state == "Rejected":
            # print("Proposer", self.unique_id, "(",self.state,") : TIME OUT...")
            self.state = "Proposing"

        # Waiting for acceptors to respond to ACCEPT message
        elif self.state == "Waiting Accept":
            total_responses = sum(self.responses)
            if total_responses >= self.model.majority :
                self.state = "Finished"
                #Send consensus to all agents
                # self.send_consensus(self.consensus)
                self.send_consensus(self.proposed_couple)

            else :
                self.state = "Rejected"


    def send_prepare(self):
        """
            Send a prepare request to all acceptors
        :return: list of responses from acceptors
        """
        #Updating proposed couple
        global last_id
        # self.consensus = (last_id, self.predicted_value)
        # self.proposed_couple = self.consensus
        self.proposed_couple  = (last_id, self.predicted_value)
        last_id += 1

        #Fetching all acceptors in the model
        acceptors = [agent for agent in self.model.schedule.agents if isinstance(agent, Acceptor)]

        #Gathering response from all acceptors
        # responses = [acceptor.receive_prepare(self.consensus) for acceptor in acceptors]
        responses = [acceptor.receive_prepare(self.proposed_couple) for acceptor in acceptors]
        return responses

    def send_accept_request(self) -> list :
        """
            Send an accept request to all acceptors
        :return: list of responses from acceptors
        """
        # print("Proposer", self.unique_id, "(",self.state,") : SEND ACCEPT REQUEST : ", self.proposed_couple, "sent.")
        acceptors = [agent for agent in self.model.schedule.agents if isinstance(agent, Acceptor)]

        # responses = [acceptor.receive_accept_request(self.consensus) for acceptor in acceptors]
        responses = [acceptor.receive_accept_request(self.proposed_couple) for acceptor in acceptors]

        return responses

    def send_consensus(self, proposal : tuple) -> None:
        """
            Send consensus to all agents
        :param proposal: the couple agreed upon
        :return: None
        """
        # print("Proposer", self.unique_id, "(",self.state,") : SEND CONSENSUS : ", proposal)
        #Send proposal to every agent
        agents = self.model.schedule.agents
        for agent in agents:
            agent.receive_consensus(proposal)

    def __str__(self):
        return f"Proposer {self.unique_id} : {self.proposed_couple} ({self.state}) {self.predicted_value}"

class Acceptor(Paxos_Predictive_Agent):
    """
        An agent that accept values proposed by proposers
    """

    def __init__(self, unique_id, model, predicted_value = None, classifier = None):
        super().__init__(unique_id, model, "Waiting", (None, None), predicted_value, classifier)

        # Whether the agent is currently broken or not
        self.broken = False
        # Time left before the agent is back
        self.time_breakdown = randint(1, 3)

    def step(self):

        if self.broken:
            self.time_breakdown -= 1
            if self.time_breakdown == 0:
                self.broken = False
                self.time_breakdown = randint(1, 3)
                # print("Acceptor", self.unique_id, "is back!")

                # for agent in self.model.schedule.agents:
                #     print(f"\t{agent}")
                #Get learner agent that is finished
                agent = [agent for agent in self.model.schedule.agents if agent.state == "Finished"]
                if len(agent) > 0:
                    self.receive_consensus(agent[0].consensus)

            return
        else :
            if self.state!="Finished" and random() < p_breakdown:
                self.broken = True
                self.model.nb_crashes += 1
                # print("Acceptor", self.unique_id, "broke down for", self.time_breakdown, "steps!")


    def receive_prepare(self, proposal) :
        """
            Receive a prepare request from a proposer
        :param proposal: the couple proposed by the proposer
        :return: True if the proposal is accepted, False otherwise
        """
        if self.broken:
            return False

        if proposal[1] == self.predicted_value and (self.consensus[0] is None or self.consensus[0] < proposal[0]):
            self.state = "Promised"
            return True
        else :
            return False

    def receive_accept_request(self, proposal) :
        """
        Response to accept request
        :param proposal: the couple proposed
        :return: True if accepted, False otherwise
        """
        if self.broken:
            return False

        if proposal[1] == self.predicted_value and (self.consensus[0] is None or self.consensus[0] <= proposal[0]):
            self.state = "Accepted"
            self.consensus = proposal
            return True
        else:
            return False

    def receive_consensus(self, proposal):
        """
        Receive consensus from other agents
        :param proposal: the couple agreed upon
        :return: None
        """
        if not self.broken:
            super().receive_consensus(proposal)

    def __str__(self):
        return f"Acceptor {self.unique_id} : {self.consensus} ({self.state}) {self.predicted_value}"

class Learner(Paxos_Agent):
    """ An agent learning from other agents"""

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model, "Waiting", (None, None))

    def __str__(self):
        return f"Learner {self.unique_id} : {self.consensus} ({self.state})"