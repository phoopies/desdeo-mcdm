from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from desdeo_mcdm.interactive.InteractiveMethod import InteractiveMethod
from desdeo_tools.interaction.request import BaseRequest, SimplePlotRequest
from desdeo_tools.scalarization.ASF import PointMethodASF
from desdeo_tools.scalarization.Scalarizer import DiscreteScalarizer
from desdeo_tools.solver.ScalarSolver import DiscreteMinimizer

"""
Nonconvex Pareto Navigator (NPN)
"""

class NPNClassificationRequest(BaseRequest): # TODO more request classes: Nimbus vs NautilusNavigator
    """
    Request to handle classifications of objectives in Nonconvex Pareto Navigator.

    Args: 
        aspiration_levels (np.ndarray): A list of aspiration levels for each objective
        upper_bounds (np.ndarray): A list of upper bounds for each objective
        navigation_point (np.ndarray): Current navigation point
    """

    def __init__( # TODO What else
        self,
        aspiration_levels: np.ndarray,
        upper_bounds: np.ndarray,
        navigation_point: np.ndarray,
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
        super().__init__() # TODO
    
    @classmethod
    def init_with_method(cls, method):
        return cls(
            # Same stuff as in __init__ 
            method.aspiration_levels
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

        # TODO the actual validations
        return

    @BaseRequest.response.setter
    def response(self, response: Dict):
        self.validator(response)
        self._response = response

class NPNException(Exception):
    """
    Raised when an exception related to Nonconvex Pareto Navigator (NPN) is encountered. 
    """

    pass

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

    def __init_(
        self,
        pareto_front: np.ndarray,
        ideal: np.ndarray,
        nadir: np.ndarray,
        objective_names: Optional[List[str]] = None,
        minimize: Optional[List[int]] = None,
    ):
        return
    
    def iterate(
        self,
        request: Union[ # Others maybe
            NPNClassificationRequest
        ],
    ) -> Tuple[Union[NPNClassificationRequest], Union[SimplePlotRequest, None]]:
        req = self.handle_request(request)
        return req
    
    def handle_request(
        self,
        request: Union[ # Others maybe
            NPNClassificationRequest
        ],
    ):
        return

    
    def solve_asf_problem(
        self,pareto_front: np.ndarray,
        ref_point: np.ndarray,
        ideal: np.ndarray,
        nadir: np.ndarray
    ) -> int:
        """Forms and solves the achievement scalarizing function to find the
        closest point on the Pareto optimal front to the given reference
        point.

        Args:
            pareto_front (np.ndarray): Approximation of the pareto front
            ref_point (np.ndarray): The reference point indicating a decision
            maker's preference.
            ideal (np.ndarray): Ideal point.
            nadir (np.ndarray): Nadir point.
        
        Returns:
            ???
        """
        asf = PointMethodASF(nadir, ideal)
        scalarizer = DiscreteScalarizer(asf, {"reference_point": ref_point})
        solver = DiscreteMinimizer(scalarizer)

        res = solver.minimize(pareto_front)

        return res
    
    def construct_navigation_set(
        self,
        pareto_outcome: np.ndarray,
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
    