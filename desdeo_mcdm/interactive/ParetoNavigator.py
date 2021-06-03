from desdeo_problem import Constraint
from desdeo_problem.Problem import MOProblem, DiscreteDataProblem
from desdeo_tools.scalarization.ASF import SimpleASF
from desdeo_tools.scalarization.Scalarizer import Scalarizer, DiscreteScalarizer
from desdeo_tools.solver.ScalarSolver import (
    ScalarMethod,
    ScalarMinimizer,
    DiscreteMinimizer,
)
from scipy.optimize.zeros import toms748
from desdeo_mcdm.interactive.ReferencePointMethod import validate_reference_point
from typing import Callable, Dict, List, Optional, Tuple, Union
from desdeo_mcdm.interactive.InteractiveMethod import InteractiveMethod
from desdeo_tools.interaction.request import BaseRequest, SimplePlotRequest
from scipy.spatial import ConvexHull
import numpy as np
import pandas as pd
from scipy.optimize import linprog

# TODO
# If maximizing then ideal > nadir -> scipy linalg fails
#   Propably because bounds will be flipped then


# Maybe DONE
# Discrete case
# Plot requests


class ParetoNavigatorException(Exception):
    """Raised when an exception related to Pareto Navigator is encountered."""

    pass


class ParetoNavigatorInitialRequest(BaseRequest):
    """
    A request class to handle the Decision Maker's initial preferences for the first iteration round.

    In what follows, the DM is involved. First, the DM is asked to select a starting
    point for the navigation phase.
    """

    def __init__(
        self,
        ideal: np.ndarray,
        nadir: np.ndarray,
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

        self._ideal = ideal
        self._nadir = nadir
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
            "ideal": ideal,
            "nadir": nadir,
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
            method._ideal,
            method._nadir,
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

        if "reference_point" in response and "preferred_solution" in response:
            msg = "Please specify only one preference method"
            raise ParetoNavigatorException(msg)

        if "reference_point" in response:
            validate_reference_point(
                response["reference_point"], self._ideal, self._nadir
            )
        elif "preferred_solution" in response:
            # Validate
            pass
        else:
            msg = "Please specify either a starting point as 'preferred_solution'."
            "or a reference point as 'reference_point."
            raise ParetoNavigatorException(msg)

        if "speed" not in response:
            msg = "Please specify a speed as 'speed'"
            raise ParetoNavigatorException(msg)

        speed = response["speed"]
        try:
            if int(speed) not in self._allowed_speeds:
                raise ParetoNavigatorException(f"Invalid speed: {speed}.")
        except Exception as e:
            raise ParetoNavigatorException(
                f"An exception rose when validating the given speed {speed}.\n"
                f"Previous exception: {type(e)}: {str(e)}."
            )

        self._response = response


class ParetoNavigatorRequest(BaseRequest):
    """
    A request class to handle navigation preferences after the first iteration round.
    """

    def __init__(
        self,
        current_solution: np.ndarray,
        ideal: np.ndarray,
        nadir: np.ndarray,
        allowed_speeds: np.ndarray,
        valid_classifications: np.ndarray,
    ):
        """
        Initialize request with current iterations's solution process information.

        Args:
            current_solution (np.ndarray): Current solution.
            ideal (np.ndarray): Ideal vector.
            nadir (np.ndarray): Nadir vector.
            allowed_speeds (np.ndarray): Allowed movement speeds
            valid_classifications (np.ndarray): Valid classifications
        """

        self._current_solution = current_solution
        self._ideal = ideal
        self._nadir = nadir
        self._allowed_speeds = allowed_speeds
        self._valid_classifications = valid_classifications

        min_speed = np.min(self._allowed_speeds)
        max_speed = np.max(self._allowed_speeds)

        msg = (
            "If you wish to see an actual pareto optimal solution based on this approximation "
            "please state 'show_solution' as 'True'. "
            "If you wish to change direction. Please specify either a "
            "new reference point as 'reference_point' or "
            "a classification for each objective as 'classification'. "
            "'classification' should be a list of strings. "
            "If you wish to step back specify 'step_back' as 'True' "
            "If you wish to change the speed, please specify a speed "
            f"as 'speed'. Speed should be an integer value between {min_speed} and {max_speed}, "
            f"where {min_speed} is the slowest speed and {max_speed} the fastest."
        )

        content = {
            "message": msg,
            "current_solution": current_solution,
            "valid_classificatons": valid_classifications,
        }

        super().__init__("reference_point_preference", "required", content=content)

    @classmethod
    def init_with_method(cls, method: InteractiveMethod):
        """
        Initialize request with given instance of ParetoNavigator.

        Args:
            method (ParetoNavigator): Instance of ParetoNavigator-class.
        Returns:
            ParetoNavigatorRequest: Initial request.
        """

        return cls(
            method._current_solution,
            method._ideal,
            method._nadir,
            method._allowed_speeds,
            method._valid_classifications,
        )

    @BaseRequest.response.setter
    def response(self, response: Dict) -> None:
        """
        Set the Decision Maker's response information for request.

        Args:
            response (Dict): The Decision Maker's response.

        Raises:
            ParetoNavigatorException: In case response is invalid.
        """

        if "show_solution" in response and response["show_solution"]:
            self._response = response
            return  # No need to validate others

        if "speed" in response:
            speed = response["speed"]
            try:
                if int(speed) not in self._allowed_speeds:
                    raise ParetoNavigatorException(f"Invalid speed: {speed}.")
            except Exception as e:
                raise ParetoNavigatorException(
                    f"An exception rose when validating the given speed {speed}.\n"
                    f"Previous exception: {type(e)}: {str(e)}."
                )

        if "reference_point" in response and "classification" in response:
            msg = (
                "Please specify only one kind of preference info if changing direction"
            )
            raise ParetoNavigatorException(msg)
        elif "reference_point" in response:
            validate_reference_point(
                response["reference_point"], self._ideal, self._nadir
            )
        elif "classification" in response:
            classifications = np.unique(response["classification"])
            if not np.all(np.isin(classifications, self._valid_classifications)):
                msg = "Invalid classifications"
                raise ParetoNavigatorException(msg)

        self._response = response


class ParetoNavigatorSolutionRequest(BaseRequest):
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


class ParetoNavigatorStopRequest(BaseRequest):
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


class ParetoNavigator(InteractiveMethod):
    """
    Paretonavigator as described in 'Pareto navigator for interactive nonlinear
    multiobjective optimization' (2008) [Petri Eskelinen · Kaisa Miettinen ·
    Kathrin Klamroth · Jussi Hakanen].

    Args:
        problem (MOProblem): The problem to be solved.
        pareto_optimal_solutions (np.ndarray): Some pareto optimal solutions to construct the polyhedral set
    """

    def __init__(
        self,
        problem: Union[MOProblem, DiscreteDataProblem],
        pareto_optimal_solutions: np.ndarray,  # Initial pareto optimal solutions
        scalar_method: Optional[ScalarMethod] = None,
    ):
        self._scalar_method = scalar_method  # CHECK

        if np.any(np.isinf(problem.nadir)) or np.any(np.isinf(problem.ideal)):
            # Get the ideal and nadir from the provided po solutions
            nadir = np.max(
                pareto_optimal_solutions, axis=0
            )  # Does it hold that ideal < nadir
            ideal = np.min(
                pareto_optimal_solutions, axis=0
            )  # if obj to maximize, does it still hold, are the values given that
            # flip if maximize
            # max_multiplier = problem._max_multiplier
            # ideal = max_multiplier*ideal
            # nadir = max_multiplier*nadir
        else:
            nadir = problem.nadir
            ideal = problem.ideal

        self._problem = problem

        self._ideal = ideal
        self._nadir = nadir
        self._n_objectives = self._ideal.shape[0]

        A, self.b = self.polyhedral_set_eq(
            pareto_optimal_solutions
        )  # Get ideal and nadir from here?
        self._weights = self.calculate_weights(self._ideal, self._nadir)

        self.lppp_A = self.construct_lppp_A(
            self._weights, A
        )  # Used in (3). Only changes if new solutions added

        self._pareto_optimal_solutions = pareto_optimal_solutions

        self._allowed_speeds = [1, 2, 3, 4, 5]

        # Improve, degrade, maintain,
        self._valid_classifications = ["<", ">", "="]

        self._current_speed = None
        self._reference_point = None
        self._current_solution = None
        self._direction = None

    def start(self):
        """
        Start the solving process

        Returns:
            ParetoNavigatorInitialRequest: Initial request
        """
        return ParetoNavigatorInitialRequest.init_with_method(self), None

    def iterate(
        self,
        request: Union[
            ParetoNavigatorInitialRequest,
            ParetoNavigatorRequest,
            ParetoNavigatorSolutionRequest,
            ParetoNavigatorStopRequest,
        ],
    ) -> Union[
        ParetoNavigatorRequest,
        ParetoNavigatorSolutionRequest,
        ParetoNavigatorStopRequest,
    ]:
        """
        Perform the next logical iteration step based on the given request type.

        Args:
            request (Union[ParetoNavigatorInitialRequest, ParetoNavigatorRequest,
            ParetoNavigatorSolutionRequest, ParetoNavigatorStopRequest]):
            A ParetoNavigatorRequest

        Returns:
            Union[ParetoNavigatorRequest, ParetoNavigatorSolutionRequest, ParetoNavigatorStopRequest]:
            A new request with content depending on the Decision Maker's preferences.
        """

        if type(request) is ParetoNavigatorInitialRequest:
            return self.handle_initial_request(request)
        elif type(request) is ParetoNavigatorRequest:
            return self.handle_request(request)
        elif type(request) is ParetoNavigatorSolutionRequest:
            return self.handle_solution_request(request)
        else:
            # if stop request, do nothing
            return request

    def handle_initial_request(
        self, request: ParetoNavigatorInitialRequest
    ) -> ParetoNavigatorRequest:
        """
        Handles the initial request.

        Args:
           request (ParetoNavigatorInitialRequest): Initial request

        Returns:
            Tuple[ParetoNavigatorRequest, SimplePlotRequest]: A navigation request and a plot request
        """
        if "reference_point" in request.response:
            self._reference_point = request.response["reference_point"]
            starting_point = self.solve_asf(self._problem, self._reference_point)
        else:  # Preferred po solution
            starting_point = self._pareto_optimal_solutions[
                request.response["preferred_solution"]
            ]

        self._current_solution = starting_point
        self._current_speed = request.response["speed"] / np.max(self._allowed_speeds)

        msg = "Current solution"
        plot_request = self.create_plot_request(
            np.atleast_2d(self._current_solution), msg
        )

        return ParetoNavigatorRequest.init_with_method(self), plot_request

    def handle_request(
        self, request: ParetoNavigatorRequest
    ) -> Tuple[
        Union[
            ParetoNavigatorRequest,
            ParetoNavigatorSolutionRequest,
            ParetoNavigatorStopRequest,
        ],
        SimplePlotRequest,
    ]:
        """
        Handles a navigation request.

        Args:
           request (ParetoNavigatorRequest): A request

        Returns:
            Tuple[
                Union[ParetoNavigatorRequest, ParetoNavigatorSolutionRequest, ParetoNavigatorStopRequest],
                SimplePlotRequest
            ]: Next request corresponding to the DM's preferences and a plot request
        """
        resp: dict = request.response
        if resp is None:
            resp = {}

        if "show_solution" in resp and resp["show_solution"]:
            self._po_solution = self.solve_asf(
                self._problem,
                self._current_solution,
            )
            self._po_objectives = self._problem.evaluate(
                self._po_solution
            ).objectives.squeeze()
            msg = "Pareto optimal solution"
            plot_request = self.create_plot_request(
                np.atleast_2d(self._po_objectives), msg
            )
            return (
                ParetoNavigatorSolutionRequest(
                    self._current_solution, self._po_solution, self._po_objectives
                ),
                plot_request,
            )

        if "speed" in resp:
            self._current_speed = resp["speed"] / np.max(self._allowed_speeds)

        if "step_back" in resp and resp["step_back"]:  # Check!
            self._current_speed *= -1
        else:  # Make sure speed is positive
            self._current_speed = np.abs(self._current_speed)

        if "reference_point" in resp:
            self._reference_point = resp["reference_point"]
        elif "classification" in resp:
            ref_point = self.classification_to_ref_point(
                resp["classification"],
                self._ideal,
                self._nadir,
                self._current_solution,
            )
            self._reference_point = ref_point

        self._direction = self.calculate_direction(
            self._current_solution, self._reference_point
        )

        # Get the new solution by solving the linear parametric problem
        self._current_solution = self.solve_linear_parametric_problem(
            self._current_solution,
            self._ideal,
            self._nadir,
            self._direction,
            self._current_speed,
            self.lppp_A,
            self.b,
        )

        msg = "Current solution"
        plot_request = self.create_plot_request(
            np.atleast_2d(self._current_solution), msg
        )
        return ParetoNavigatorRequest.init_with_method(self), plot_request

    def handle_solution_request(
        self, request: ParetoNavigatorSolutionRequest
    ) -> Union[ParetoNavigatorRequest, ParetoNavigatorStopRequest]:
        """
        Handle a solution request

        Args:
            request (ParetoNavigatorSolutionRequest): A solution request

        Returns:
            Tuple[
                Union[ParetoNavigatorRequest, ParetoNavigatorStopRequest],
                SimplePlotRequest
            ]: A navigation request or a stop request depending on whether
            the DM wishes to continue or stop and a plot request
        """
        resp = request.response
        if resp is not None:  # Is not satisfied with the solution
            if "satisfied" in resp and resp["satisfied"]:
                final_solution = self._po_solution
                msg = "Final solution reached"
                plot_request = self.create_plot_request(
                    np.atleast_2d(self._po_objectives), msg
                )
                stop_request = ParetoNavigatorStopRequest(
                    self._current_solution, final_solution, self._po_objectives
                )
                return stop_request, plot_request

        # No response or not satisfied

        # Add solution to approximation
        self._pareto_optimal_solutions = np.vstack(
            (self._pareto_optimal_solutions, self._po_objectives)
        )
        A, self.b = self.polyhedral_set_eq(self._pareto_optimal_solutions)
        # Update ideal and nadir, then also weights...
        # self._weights = self.calculate_weights(self._ideal, self._nadir)

        # update lppp A
        self.lppp_A = self.construct_lppp_A(self._weights, A)

        msg = "Previous solution"
        plot_request = self.create_plot_request(np.atleast_2d(self._po_objectives), msg)

        return ParetoNavigatorRequest.init_with_method(self), plot_request

    def create_plot_request(
        self, objectives: np.ndarray, msg: str
    ) -> SimplePlotRequest:
        """Used to create a plot request for visualizing objective values.

        Args:
            objectives (np.ndarray): A 2D numpy array containing objective vectors to be visualized.
            msg (str): A message to be displayed in the context of a visualization.

        Returns:
            SimplePlotRequest: A plot request to create a visualization.
        """
        if isinstance(self._problem, MOProblem):
            dimensions_data = pd.DataFrame(
                index=["minimize", "ideal", "nadir"],
                columns=self._problem.get_objective_names(),
            )
            dimensions_data.loc["minimize"] = self._problem._max_multiplier
            dimensions_data.loc["ideal"] = self._ideal
            dimensions_data.loc["nadir"] = self._nadir

            data = pd.DataFrame(objectives, columns=self._problem.get_objective_names())
        else:
            dimensions_data = pd.DataFrame(
                index=["minimize", "ideal", "nadir"],
                columns=self._problem.objective_names,
            )
            dimensions_data.loc["minimize"] = [1 for _ in self._problem.objective_names]
            dimensions_data.loc["ideal"] = self._ideal
            dimensions_data.loc["nadir"] = self._nadir

            data = pd.DataFrame(objectives, columns=self._problem.objective_names)

        plot_request = SimplePlotRequest(
            data=data,
            dimensions_data=dimensions_data,
            message=msg,
        )

        return plot_request

    def calculate_weights(self, ideal: np.ndarray, nadir: np.ndarray):
        """
        Calculate the scaling coefficients w from ideal and nadir.

        Args:
            ideal (np.ndarray): Ideal vector
            nadir (np.ndarray): Nadir vector

        Returns:
            np.ndarray: The scaling coefficients
        """
        return 1 / (nadir - ideal)

    def polyhedral_set_eq(
        self, po_solutions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Construct a polyhedral set as convex hull
        from the set of pareto optimal solutions

        Args:
            po_solutions (np.ndarray): Some pareto optimal solutions

        Returns:
            (np.ndarray, np.ndarray): Matrix A and vector b from the
            convex hull inequality representation Az <= b
        """
        convex_hull = ConvexHull(po_solutions)  # facet: Az + b = 0 so inside: Az <= -b
        A = convex_hull.equations[:, 0:-1]
        b = -convex_hull.equations[:, -1]
        return A, b

    def construct_lppp_A(self, weights, A):
        """
        The matrix A used in the linear parametric programming problem

        Args:
            weights (np.ndarray): Scaling coefficients
            A (np.ndarray): Matrix A from the convex hull representation Ax < b

        Returns:
            np.ndarray: The matrix A' in the linear parametric programming problem A'x<b'
        """
        k = len(weights)
        diag = np.zeros((k, k))

        np.fill_diagonal(diag, 1)
        weights_inv = np.reshape(np.vectorize(lambda w: -1 / w)(weights), (k, 1))
        upper_A = np.hstack((weights_inv, diag))

        fill_zeros = np.zeros((len(A), 1))
        filled_A = np.hstack((fill_zeros, A))

        lppp_A = np.concatenate((upper_A, filled_A))
        return lppp_A

    def calculate_direction(self, current_solution: np.ndarray, ref_point: np.ndarray):
        """
        Calculate a new direction from current solution and a given reference point

        Args:
            current_solution (np.ndarray): The current solution
            ref_point (np.ndarray): A reference point

        Returns:
            np.ndarray: A new direction
        """
        return ref_point - current_solution

    def classification_to_ref_point(
        self, classifications, ideal, nadir, current_solution
    ):
        """
        Transform a classification to a reference point

        Args:
            classifications (np.ndarray): Classification for each objective
            ideal (np.ndarray): Ideal point
            nadir (np.ndarray): Nadir point
            current_solution (np.ndarray): Current solution

        Returns:
            np.ndarray: A reference point which is constructed from the classifications
        """

        def mapper(c: str, i: int):
            if c == "<":
                return ideal[i]
            elif c == ">":
                return nadir[i]
            elif c == "=":
                return current_solution[i]

        ref_point = [mapper(c, i) for i, c in (list(enumerate(classifications)))]
        return np.array(ref_point)

    def solve_linear_parametric_problem(
        self,
        current_sol: np.ndarray,
        ideal: np.ndarray,
        nadir: np.ndarray,
        direction: np.ndarray,
        a: float,
        A: np.ndarray,
        b: np.ndarray,
    ) -> np.ndarray:
        """
        Solves the linear parametric programming problem
        as defined in (3)

        Args:
            current_sol (np.ndarray): Current solution
            ideal (np.ndarray): Ideal vector
            nadir (np.ndarray): Nadir vector
            direction (np.ndarray): Navigation direction
            a (float): Alpha in problem (3)
            A (np.ndarray): Matrix A from Az <= b
            b (np.ndarray): Vector b from Az <= b

        Returns:
            np.ndarray: Optimal vector from the linear parametric programming problem.
            This is the new solution to be used in the navigation.
        """
        k = len(current_sol)
        c = np.array([1] + k * [0])

        moved_ref_point = current_sol + (a * direction)
        moved_ref_point = np.reshape(moved_ref_point, ((k, 1)))
        b_new = np.append(moved_ref_point, b)

        obj_bounds = np.stack((ideal, nadir))
        bounds = [(None, None)] + [(x, y) for x, y in obj_bounds.T]
        sol = linprog(c=c, A_ub=A, b_ub=b_new, bounds=bounds)
        if sol["success"]:
            return sol["x"][1:]  # zeta in index 0.
        else:
            raise ParetoNavigatorException("Couldn't calculate a new solution")

    def solve_asf(
        self, problem: Union[MOProblem, DiscreteDataProblem], ref_point: np.ndarray
    ):
        """
        Solve achievement scalarizing function with simpleasf

        Args:
            problem (MOProblem): The problem
            ref_point: A reference point

        Returns:
            np.ndarray: The decision vector which solves the achievement scalarizing function
        """
        asf = SimpleASF(np.ones(ref_point.shape))
        if isinstance(problem, MOProblem):
            scalarizer = Scalarizer(
                lambda x: problem.evaluate(x).objectives,
                asf,
                scalarizer_args={"reference_point": np.atleast_2d(ref_point)},
            )

            if problem.n_of_constraints > 0:
                _con_eval = lambda x: problem.evaluate(x).constraints.squeeze()
            else:
                _con_eval = None

            solver = ScalarMinimizer(
                scalarizer,
                problem.get_variable_bounds(),
                constraint_evaluator=_con_eval,
                method=self._scalar_method,
            )

            res = solver.minimize(problem.get_variable_upper_bounds() / 2)
        else:  # Discrete case
            scalarizer = DiscreteScalarizer(
                asf, scalarizer_args={"reference_point": np.atleast_2d(ref_point)}
            )
            solver = DiscreteMinimizer(scalarizer)
            res = solver.minimize(problem.objectives)

        if res["success"]:
            return res["x"]
        else:
            raise ParetoNavigatorException(
                "Could solve achievement scalarazing function"
            )


# Testing
if __name__ == "__main__":
    from desdeo_problem.Objective import _ScalarObjective
    from desdeo_problem import variable_builder

    # Objectives
    def f1(xs):
        xs = np.atleast_2d(xs)
        return -xs[:, 0] - xs[:, 1] + 5

    def f2(xs):
        xs = np.atleast_2d(xs)
        return (1 / 5) * (
            np.square(xs[:, 0])
            - 10 * xs[:, 0]
            + np.square(xs[:, 1])
            - 4 * xs[:, 1]
            + 11
        )

    def f3(xs):
        xs = np.atleast_2d(xs)
        return (5 - xs[:, 0]) * (xs[:, 1] - 11)

    obj1 = _ScalarObjective("obj1", f1)
    obj2 = _ScalarObjective("obj2", f2)
    obj3 = _ScalarObjective("obj3", f3)
    objectives = [obj1, obj2, obj3]
    objectives_n = len(objectives)

    # variables
    var_names = ["x1", "x2"]
    variables_n = len(var_names)

    initial_values = np.array([2, 3])
    lower_bounds = [0, 0]
    upper_bounds = [4, 6]
    bounds = np.stack((lower_bounds, upper_bounds))
    variables = variable_builder(var_names, initial_values, lower_bounds, upper_bounds)

    # Constraints
    def c1(xs, ys):
        xs = np.atleast_2d(xs)
        return np.negative((3 * xs[:, 0] + xs[:, 1] - 12))

    def c2(xs, ys):
        xs = np.atleast_2d(xs)
        return np.negative((2 * xs[:, 0] + xs[:, 1] - 9))

    def c3(xs, ys):
        xs = np.atleast_2d(xs)
        return np.negative((xs[:, 0] + 2 * xs[:, 1] - 12))

    con1 = Constraint.ScalarConstraint("c1", variables_n, objectives_n, c1)
    con2 = Constraint.ScalarConstraint("c2", variables_n, objectives_n, c2)
    con3 = Constraint.ScalarConstraint("c3", variables_n, objectives_n, c3)
    constraints = [con1, con2, con3]

    # problem
    problem = MOProblem(
        objectives=objectives, variables=variables, constraints=constraints
    )

    po_sols = np.array(
        [
            [-2, 0, -18],
            [-1, 4.6, -25],
            [0, -3.1, -14.25],
            [1.38, 0.62, -35.33],
            [1.73, 1.72, -38.64],
            [2.48, 1.45, -42.41],
            [5.00, 2.20, -55.00],
        ]
    )

    method = ParetoNavigator(problem, po_sols)
    print(method._ideal, method._nadir)

    request, plot_req = method.start()
    print(request.content["message"])

    request.response = {
        "preferred_solution": 3,
        "speed": 2,
    }

    request, plot_req = method.iterate(request)
    print(request.content["message"])
    print(request.content["current_solution"])

    request.response = {
        # 'reference_point': np.array([ideal[0], ideal[1], nadir[2]]),
        "classification": ["<", "<", ">"],
    }

    for i in range(50):
        request, plot_req = method.iterate(request)
        cur_sol = request.content["current_solution"]
        print(cur_sol)
        if np.all(np.abs(cur_sol - np.array([0.35, -0.51, -26.26])) < 0.3):
            break

    request.response = {
        "classification": ["<", ">", "="],
        #'reference_point': np.array([ideal[0], nadir[1], cur_sol[2]]),
    }

    for i in range(50):
        request, plot_req = method.iterate(request)
        cur_sol = request.content["current_solution"]
        print(cur_sol)
        if np.all(np.abs(cur_sol - np.array([-0.64, 1.82, -25.95])) < 0.3):
            break

    request.response = {
        "classification": [">", "<", "<"],
    }

    for i in range(50):
        request, plot_req = method.iterate(request)
        cur_sol = request.content["current_solution"]
        print(cur_sol)
        if np.all(np.abs(cur_sol - np.array([-0.32, 2.33, -27.85])) < 0.3):
            break

    request.response = {
        "show_solution": True,
    }

    request, plot_req = method.iterate(request)
    print(request.content["message"])
    print(request.content["objective_values"])

    request, plot_req = method.iterate(request)  # Not satisfied

    print(request.content["message"])

    request.response = {
        "classification": ["<", ">", ">"],
        "speed": 1,
    }

    for i in range(4):
        request, plot_req = method.iterate(request)
        cur_sol = request.content["current_solution"]
        print(cur_sol)

    request.response = {
        "show_solution": True,
    }

    request, plot_req = method.iterate(request)
    request.response = {"satisfied": True}
    request, plot_req = method.iterate(request)

    print(request.content["message"])
    print(request.content["final_solution"])
    obj_values = request.content["objective_values"]
    print(obj_values)

    import matplotlib.pyplot as plt

    ax = plt.axes(projection='3d')
    
    # Scatter pareto optimal solutions
    xs, ys, zs = np.hsplit(method._pareto_optimal_solutions, 3)
    ax.scatter(xs, ys, zs)

    # Scatter Pareto Optimal solutions received by navigation
    x, y, z = np.hsplit(obj_values, 3)
    ax.scatter(x, y, z, marker="o")

    convex_hull = ConvexHull(method._pareto_optimal_solutions)

    #Plotting the convex hull
    from matplotlib.patches import Polygon
    for s in convex_hull.simplices:
        s = np.append(s, s[0])  # Here we cycle back to the first coordinate
        ax.plot(
            method._pareto_optimal_solutions[s, 0], 
            method._pareto_optimal_solutions[s, 1],
            method._pareto_optimal_solutions[s, 2], 
            "r-"
        )

    plt.show()