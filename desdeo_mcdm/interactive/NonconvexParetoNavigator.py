from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from desdeo_problem.Problem import MOProblem, DiscreteDataProblem
from desdeo_mcdm.interactive.InteractiveMethod import InteractiveMethod
from desdeo_tools.interaction.request import BaseRequest, SimplePlotRequest
from desdeo_tools.scalarization.ASF import PointMethodASF
from desdeo_tools.scalarization.Scalarizer import DiscreteScalarizer
from desdeo_tools.solver.ScalarSolver import DiscreteMinimizer

"""
Nonconvex Pareto Navigator (NPN)
"""


class NPNException(Exception):
    """
    Raised when an exception related to Nonconvex Pareto Navigator (NPN) is encountered. 
    """

    pass


class NPNInitialRequest(BaseRequest):
    """
    A request class to handle the Decision Maker's initial preferences for the first iteration round.

    In what follows, the DM is involved. First, the DM is asked to select a starting
    point for the navigation phase.
    """

    def __init__(
        self,
        # ideal: np.ndarray,
        # nadir: np.ndarray,
        allowed_speeds: np.ndarray,
        po_solutions: np.ndarray,
    ):
        """
        Args:
            ideal (np.ndarray): Ideal vector
            nadir (np.ndarray): Nadir vector
            allowed_speeds (np.ndarray): Allowed movement speeds
            po_solutions: (np.ndarray): A small set of pareto optimal solutions
        """

        self._allowed_speeds = allowed_speeds
        self._po_solutions = po_solutions

        min_speed = np.min(self._allowed_speeds)
        max_speed = np.max(self._allowed_speeds)

        msg = "Please specify a starting point as 'preferred_solution'."
        "Or specify a reference point as 'reference_point'."
        "Please specify speed as 'speed' to be an integer value between"
        f"{min_speed} and {max_speed} "
        f"where {min_speed} is the slowest speed and {max_speed} the fastest."

        content = {
            "message": msg,
            "pareto_optimal_solutions": po_solutions,
            "allowed_speeds": allowed_speeds,
        }

        super().__init__("preferred_solution_preference", "required", content=content)

    @classmethod
    def init_with_method(cls, method: InteractiveMethod):
        """
        Initialize request with given instance of ParetoNavigator.

        Args:
            method (ParetoNavigator): Instance of ReferencePointMethod-class.
        Returns:
            ParetoNavigatorInitialRequest: Initial request.
        """

        return cls(
            method._allowed_speeds,
            method._pareto_optimal_solutions,
        )

    @BaseRequest.response.setter
    def response(self, response: Dict) -> None:
        """
        Set the Decision Maker's response information for initial request.

        Args:
            response (Dict): The Decision Maker's response.

        Raises:
            ParetoNavigatorException: In case reference point
            or preferred solution is missing.
        """


        if "preferred_solution" in response:
            # Validate
            pass
        else:
            msg = "Please specify either a starting point as 'preferred_solution'."
            "or a reference point as 'reference_point."
            raise NPNException(msg)

        if "speed" not in response:
            msg = "Please specify a speed as 'speed'"
            raise NPNException(msg)

        speed = response["speed"]
        try:
            if int(speed) not in self._allowed_speeds:
                raise NPNException(f"Invalid speed: {speed}.")
        except Exception as e:
            raise NPNException(
                f"An exception rose when validating the given speed {speed}.\n"
                f"Previous exception: {type(e)}: {str(e)}."
            )

        self._response = response


class NPNClassificationRequest(BaseRequest):
    """
    Request to handle classifications of objectives in Nonconvex Pareto Navigator.

    Args: 
        aspiration_levels (np.ndarray): A list of aspiration levels for each objective
        upper_bounds (np.ndarray): A list of upper bounds for each objective
        navigation_point (np.ndarray): Current navigation point
    """

    def __init__(
        self,
        # aspiration_levels: np.ndarray,
        # upper_bounds: np.ndarray,
        # navigation_point: np.ndarray,
    ):
        msg = (
            # TODO Make a list of other things needed
            "Please supply aspirations levels as 'aspiration_levels' and upper bounds"
            "as 'upper_bounds' for each objective. If no aspiration level or upper bound is"
            "desired for an objective mark the value as None. If going to the previous step"
            "is desired, set 'go_to_previous' to True, otherwise set it to False."
            "Lastly if satisfied with the current solution, set 'stop' to True,"
            "otherwise set it to False." 
        )

        content = {
            "message": msg,

        }
        super().__init__("classification_preference", "required", content=content) # TODO
    
    @classmethod
    def init_with_method(cls, method):
        return cls(
            # TODO
        )
    
    def validator(self, response: Dict) -> None:
        """
        Validates a dictionary containing the response of the decision maker. Should contain the keys
        'aspiration_levels', and 'upper_bounds'.

        'aspiration_levels' should be a list of aspiration levels (float | None) for each objective. 
        'upper_bounds' should be a list of upper bounds (float | None) for each objective.
        The size of both of the lists should equal to the count of objectives. 
        If no aspiration level or upper bound is specified for an objective should the corresponding value
        in the list be equal to None. An objective can have both an aspiration level and an upper bound.

        Args:
            response (Dict): See documentation
        
        Raises:
            NPNException: In case the response in invalid
        """

        # TODO validations
        return

    @BaseRequest.response.setter
    def response(self, response: Dict):
        self.validator(response)
        self._response = response


class NPNSolutionRequest(BaseRequest):
    """
    A request class to handle ...
    """

    def __init__(
        self,
        approx_solution: np.ndarray,
        pareto_optimal_solution: np.ndarray,
        objective_values: np.ndarray,
    ):
        """
        Args:
            approx_solution (np.ndarray): The approximated solution received by navigation
            pareto_optimal_solution (np.ndarray): A pareto optimal solution (decision variables).
            objective_values (np.ndarray): Objective vector.
        """
        msg = (
            "If you are satisfied with this pareto optimal solution "
            "please state 'satisfied' as 'True'. This will end the navigation. "
            "Otherwise state 'satisfied' as 'False and the navigation will "
            "be continued with this pareto optimal solution added to the approximation."
        )

        content = {
            "message": msg,
            "approximate_solution": approx_solution,
            "pareto_optimal_solution": pareto_optimal_solution,
            "objective_values": objective_values,
        }

        super().__init__("preferred_solution_preference", "required", content=content)

    @BaseRequest.response.setter
    def response(self, response: Dict) -> None:
        """
        Set the Decision Maker's response information for request.

        Args:
            response (Dict): The Decision Maker's response.
        """
        self._response = response


class NPNStopRequest(BaseRequest):
    """
    A request class to handle termination.
    """

    def __init__(
        self,
        approx_solution: np.ndarray,
        final_solution: np.ndarray,
        objective_values: np.ndarray,
    ):
        """
        Initialize termination request with approximate solution,
        final solution and corresponding objective vector.

        Args:
            approx_solution (np.ndarray): The approximated solution received by navigation
            final_solution (np.ndarray): Solution (decision variables).
            objective_values (np.ndarray): Objective vector.
        """
        msg = "Final solution found."

        content = {
            "message": msg,
            "approximate_solution": approx_solution,
            "final_solution": final_solution,
            "objective_values": objective_values,
        }

        super().__init__("print", "no_interaction", content=content)



class NonconvexParetoNavigator(InteractiveMethod):
    """
    Implements the Nonconvex Pareto Navigator as presented in 
    |Interactive Nonconvex Pareto Navigator for multiobjective optimization 2019|

    Args:
        pareto_front (np.ndarray): A two dimensional numpy array
        representing a Pareto front with objective vectors on each of its
        rows. Will be used to generate an approximation of the Pareto optimal front
        in the objective space with the PAINT method.
        ideal (np.ndarray): The ideal objective vector of the problem
        being represented by the Pareto front.
        nadir (np.ndarray): The nadir objective vector of the problem
        being represented by the Pareto front.
        objective_names (Optional[List[str]], optional): Names of the
        objectives.
        minimize (Optional[List[int]], optional): Multipliers for each
        objective. '-1' indicates maximization and '1' minimization.
        Defaults to all objective values being minimized.

    Raises:
        NPNException: A dimension mismatch encountered among supplied arguments
    """

    def __init__(
        self,
        problem: Union[MOProblem, DiscreteDataProblem],
        pareto_front: np.ndarray,
    ):
        self._problem = problem
        self._pareto_front = pareto_front

        self._speed = None
        self._navigation_point = None
        self._reference_point = None

        self._allowed_speeds = [1,2,3,4,5]
        self._valid_classifications = ["<", "<=", "=", ">=", "0"]
    
    def start(self):
        return NPNInitialRequest.init_with_method(self), None
    
    def iterate(
        self,
        request: Union[
            NPNInitialRequest,
            NPNClassificationRequest,
            NPNSolutionRequest,
            NPNStopRequest,
        ],
    ) -> Tuple[Union[NPNClassificationRequest, NPNSolutionRequest], Union[SimplePlotRequest, None]]:
        """
        TODO
        """
        if type(request) is NPNInitialRequest:
            return self.handle_initial_request(request)
        elif type(request) is NPNClassificationRequest:
            return self.handle_classification_request(request)
        elif type(request) is NPNSolutionRequest:
            return self.handle_solution_request(request)
        else:
            # if stop request, do nothing
            return request

    def handle_initial_request(
        self, request: NPNInitialRequest
    ) -> NPNClassificationRequest:
        """
        TODO
        """
        pass

    def handle_classification_request(
        self,
        request: NPNClassificationRequest
    ):
        """
        TODO
        """
        pass
    
    def handle_solution_request(
        self,
        request: NPNStopRequest
    ):
        """
        TODO
        """
        pass

    
    def solve_asf_problem(
        self,pareto_front: np.ndarray,
        ref_point: np.ndarray,
        ideal: np.ndarray,
        nadir: np.ndarray
    ) -> int:
        """Solves the achievement scalarizing function

        Args:
            pareto_front (np.ndarray): Approximation of the pareto front
            ref_point (np.ndarray): The reference point indicating a decision
            maker's preference.
            ideal (np.ndarray): Ideal point.
            nadir (np.ndarray): Nadir point.
        
        Returns:
            np.ndarray: The solution from the projection
        """
        # asf = PointMethodASF(nadir, ideal)
        pass
    
    def construct_navigation_set(
        self,
        pareto_solutions: np.ndarray,
    ):
        """
        
        """
        return
    
    def paint_method(
        self
    ):
        """
        TODO this one should propably be in an other package!
        Constructs the navigation set using the PAINT method
        """
        return
    
    def construct_econe(self, eps: np.ndarray = None):
        """
        TODO this one should propably be in an other package!
        Constructs the econe B_\eps
        """
        k = self.ideal.shape[0]
        if eps is None:
            eps = np.array([1e-6] * k)
        else: #check also that e_j > 0 forall j = i..k
            if k != eps.shape[0]:
                raise NPNException(f"Epsilon component count {eps.shape[0]} should match the objective count {k}")
            if not all(e > 0 for e in eps):
                raise NPNException("A component in epsilon is not positive")
        
        v = self.construct_extreme_rays_matrix(eps)
        

    def construct_extreme_rays_matrix(self, eps: np.ndarray) -> np.ndarray:
        k = eps.shape[0]
        v = [[-eps[j] if j != i else 1 for i in range(k)] for j in range(k)]
        return v
    
    def calculate_ref_points(
        self,
        nav_point: np.ndarray,
        ideal: np.ndarray,
        eps: float,
        step_count: int,
        nav_par_ubound: int,
    ):
        ref_point = 0 # fun
        return
    
    def calculate_navigation_point(
        self,
    ):
        return





if __name__ == "__main__":
    print("Might need to finish the navigator before I can make an example")
    # TODO example/test
    