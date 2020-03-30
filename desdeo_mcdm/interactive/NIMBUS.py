import numpy as np
import pandas as pd

from typing import List, Union, Tuple

from desdeo_mcdm.interactive.InteractiveMethod import InteractiveMethod
from desdeo_mcdm.utilities.solvers import payoff_table_method
from desdeo_tools.interaction.request import BaseRequest, SimplePlotRequest
from desdeo_tools.scalarization.ASF import (
    SimpleASF,
    MaxOfTwoASF,
    StomASF,
    PointMethodASF,
    AugmentedGuessASF,
)
from desdeo_tools.solver.ScalarSolver import ScalarMinimizer
from desdeo_tools.scalarization.Scalarizer import Scalarizer
from desdeo_problem.Problem import MOProblem


class NimbusClassificationRequest(BaseRequest):
    def __init__(self, ref: np.ndarray):
        msg = (
            "Please classify each of the objective values in one of the following categories:"
            "\n\t1. values should improve '<'"
            "\n\t2. values should improve until some desired aspiration level is reached '<='"
            "\n\t3. values with an acceptable level '='"
            "\n\t4. values which may be impaired until some upper bound is reached '>='"
            "\n\t5. values which are free to change '0'"
            "\nProvide the aspiration levels and upper bounds as a vector. For categories 1, 3, and 5,"
            "the value in the vector at the objective's position is ignored. Suppy also the number of maximum"
            "solutions to be generated."
        )
        content = {
            "message": msg,
            "objective_values": ref,
            "classifications": [None],
            "levels": [None],
            "number_of_solutions": 1,
        }
        super().__init__(
            "classification_preference", "required", content=content
        )


class NimbusSaveRequest(BaseRequest):
    def __init__(
        self,
        solution_vectors: List[np.ndarray],
        objective_vectors: List[np.ndarray],
    ):
        msg = (
            "Please specify which solutions shown you would like to save for later viewing. Supply the "
            "indices of such solutions as a list, or supply an empty list if none of the shown soulutions "
            "should be saved."
        )
        content = {
            "message": msg,
            "solutions": solution_vectors,
            "objectives": objective_vectors,
            "indices": [],
        }
        super().__init__(
            "classification_preference", "required", content=content
        )


class NimbusIntermediateSolutionsRequest(BaseRequest):
    pass


class NIMBUS(InteractiveMethod):
    """Implements the synchronous NIMBUS variant.
    """

    def __init__(self, problem: MOProblem):
        # check if ideal and nadir are defined
        if problem.ideal is None or problem.nadir is None:
            ideal, nadir = payoff_table_method(problem)
            self._ideal = ideal
            self._nadir = nadir
        else:
            self._ideal = problem.ideal
            self._nadir = problem.nadir

        # generate Pareto optimal starting point
        asf = SimpleASF(np.ones(self._ideal.shape))
        scalarizer = Scalarizer(
            lambda x: problem.evaluate(x).objectives,
            asf,
            scalarizer_args={"reference_point": np.atleast_2d(self._ideal)},
        )

        if problem.n_of_constraints > 0:
            _con_eval = lambda x: problem.evaluate(x).constraints.squeeze()
        else:
            _con_eval = None
        solver = ScalarMinimizer(
            scalarizer,
            problem.get_variable_bounds(),
            constraint_evaluator=_con_eval,
        )

        res = solver.minimize(problem.get_variable_upper_bounds() / 2)

        if res["success"]:
            self._current_solution = res["x"]
            self._current_objectives = problem.evaluate(
                self._current_solution
            ).objectives.squeeze()

        self._archive_solutions = []
        self._archive_objectives = []
        self._state = "classify"

        super().__init__(problem)

    def requests(self) -> List[BaseRequest]:
        if self._state == "classify":
            return [
                NimbusClassificationRequest(self._current_solution.squeeze())
            ]

    def handle_classification_request(
        self, request: NimbusClassificationRequest
    ) -> Tuple[NimbusSaveRequest, SimplePlotRequest]:
        # check the classifications
        is_valid_cls = map(
            lambda x: x in ["<", "<=", "=", ">=", "0"],
            request.response["classifications"],
        )
        if not all(list(is_valid_cls)):
            print("not fine")
        else:
            print("fine")

        # check the levels
        if (
            len(np.array(request.response["levels"]).squeeze())
            != self._problem.n_of_objectives
        ):
            print("not fine")
        else:
            print("fine")

        # check the levels have the right dimensions
        if (
            len(np.array(request.response["levels"]).squeeze())
            != self._problem.n_of_objectives
        ):
            print("bad levels")

        improve_until_inds = np.where(
            np.array(request.response["classifications"]) == "<="
        )[0]
        print(improve_until_inds)

        impaire_until_inds = np.where(
            np.array(request.response["classifications"]) == ">="
        )[0]
        print(impaire_until_inds)

        if len(improve_until_inds) > 0:
            # some objectives classified to be improved until some level
            if not np.all(
                np.array(request.response["levels"])[improve_until_inds]
                >= self._ideal[improve_until_inds]
            ) or not np.all(
                np.array(request.response["levels"])[improve_until_inds]
                <= self._nadir[improve_until_inds]
            ):
                print("bad improve levels!")
            else:
                print("fine improve levels")
        else:
            print("i'm empty")

        if len(impaire_until_inds) > 0:
            # some objectives classified to be improved until some level
            if not np.all(
                np.array(request.response["levels"])[impaire_until_inds]
                >= self._ideal[impaire_until_inds]
            ) or not np.all(
                np.array(request.response["levels"])[impaire_until_inds]
                <= self._nadir[impaire_until_inds]
            ):
                print("bad impair levels!")
            else:
                print("fine impair levels")
        else:
            print("i'm empty")

        improve_inds = np.where(
            np.array(request.response["classifications"]) == "<"
        )[0]

        acceptable_inds = np.where(
            np.array(request.response["classifications"]) == "="
        )[0]

        free_inds = np.where(
            np.array(request.response["classifications"]) == "0"
        )[0]

        # check maximum number of solutions
        if (
            request.response["number_of_solutions"] > 4
            or request.response["number_of_solutions"] < 1
        ):
            print("not fine")
        else:
            print("fine")

        # calculate the new solutions
        return self.calculate_new_solutions(
            int(request.response["number_of_solutions"]),
            np.array(request.response["levels"]),
            improve_inds,
            improve_until_inds,
            acceptable_inds,
            impaire_until_inds,
            free_inds,
        )

    def handle_save_request(
        self, request: NimbusSaveRequest
    ) -> Tuple[NimbusIntermediateSolutionsRequest, None]:
        if not request.response["indices"]:
            # nothing to save, continue to next state
            pass

        if (
            len(request.response["indices"])
            > len(request.content["objectives"])
            or len(request.response["indices"]) < 0
        ):
            # wrong number of indices
            print("ERROR 1")

        if (
            np.max(request.response["indices"])
            >= len(request.content["objectives"])
            or np.min(request.response["indices"]) < 0
        ):
            # out of bounds index
            print("ERROR 2")

        # save the solutions to the archive
        return self.save_solutions_to_archive(
            np.array(request.content["objectives"]),
            np.array(request.content["solutions"]),
            np.array(request.response["indices"]),
        )

    def save_solutions_to_archive(
        self,
        objectives: np.ndarray,
        decision_variables: np.ndarray,
        indices: List[int],
    ) -> Tuple[NimbusIntermediateSolutionsRequest, None]:
        self._archive_objectives.extend(list(objectives[indices]))
        self._archive_solutions.extend(list(decision_variables[indices]))
        # create intermediate point request
        pass

    def calculate_new_solutions(
        self,
        number_of_solutions: int,
        levels: np.ndarray,
        improve_inds: np.ndarray,
        improve_until_inds: np.ndarray,
        acceptable_inds: np.ndarray,
        impaire_until_inds: np.ndarray,
        free_inds: np.ndarray,
    ) -> Tuple[NimbusSaveRequest, SimplePlotRequest]:
        results = []

        # always computed
        asf_1 = MaxOfTwoASF(
            self._nadir, self._ideal, improve_inds, improve_until_inds
        )

        def cons_1(
            x: np.ndarray,
            f_current: np.ndarray = self._current_objectives,
            levels: np.ndarray = levels,
            improve_until_inds: np.ndarray = improve_until_inds,
            improve_inds: np.ndarray = improve_inds,
            impaire_until_inds: np.ndarray = impaire_until_inds,
        ):
            f = self._problem.evaluate(x).objectives.squeeze()

            res_1 = f_current[improve_inds] - f[improve_inds]
            res_2 = f_current[improve_until_inds] - f[improve_until_inds]
            res_3 = levels[impaire_until_inds] - f_current[impaire_until_inds]

            res = np.hstack((res_1, res_2, res_3))

            if self._problem.n_of_constraints > 0:
                res_prob = self._problem.evaluate(x).constraints.squeeze()

                return np.hstack((res_prob, res))

            else:
                return res

        scalarizer_1 = Scalarizer(
            lambda x: self._problem.evaluate(x).objectives,
            asf_1,
            scalarizer_args={"reference_point": levels},
        )

        solver_1 = ScalarMinimizer(
            scalarizer_1, self._problem.get_variable_bounds(), cons_1, None
        )

        res_1 = solver_1.minimize(self._current_solution)
        results.append(res_1)

        if number_of_solutions > 1:
            # create the reference point needed in the rest of the ASFs
            z_bar = np.zeros(self._problem.n_of_objectives)
            z_bar[improve_inds] = self._ideal[improve_inds]
            z_bar[improve_until_inds] = levels[improve_until_inds]
            z_bar[acceptable_inds] = self._current_objectives[acceptable_inds]
            z_bar[impaire_until_inds] = levels[impaire_until_inds]
            z_bar[free_inds] = self._nadir[free_inds]

            # second ASF
            asf_2 = StomASF(self._ideal)

            # cons_2 can be used in the rest of the ASF scalarizations, it's not a bug!
            if self._problem.n_of_constraints > 0:
                cons_2 = lambda x: self._problem.evaluate(
                    x
                ).constraints.squeeze()
            else:
                cons_2 = None

            scalarizer_2 = Scalarizer(
                lambda x: self._problem.evaluate(x).objectives,
                asf_2,
                scalarizer_args={"reference_point": z_bar},
            )

            solver_2 = ScalarMinimizer(
                scalarizer_2, self._problem.get_variable_bounds(), cons_2, None
            )

            res_2 = solver_2.minimize(self._current_solution)
            results.append(res_2)

        if number_of_solutions > 2:
            # asf 3
            asf_3 = PointMethodASF(self._nadir, self._ideal)

            scalarizer_3 = Scalarizer(
                lambda x: self._problem.evaluate(x).objectives,
                asf_3,
                scalarizer_args={"reference_point": z_bar},
            )

            solver_3 = ScalarMinimizer(
                scalarizer_3, self._problem.get_variable_bounds(), cons_2, None
            )

            res_3 = solver_3.minimize(self._current_solution)
            results.append(res_3)

        if number_of_solutions > 3:
            # asf 4
            asf_4 = AugmentedGuessASF(self._nadir, self._ideal, free_inds)

            scalarizer_4 = Scalarizer(
                lambda x: self._problem.evaluate(x).objectives,
                asf_4,
                scalarizer_args={"reference_point": z_bar},
            )

            solver_4 = ScalarMinimizer(
                scalarizer_4, self._problem.get_variable_bounds(), cons_2, None
            )

            res_4 = solver_4.minimize(self._current_solution)
            results.append(res_4)

        # create the save request
        solutions = [res["x"] for res in results]
        objectives = [
            self._problem.evaluate(x).objectives.squeeze() for x in solutions
        ]

        save_request = NimbusSaveRequest(solutions, objectives)

        # create the plot request
        dimensions_data = pd.DataFrame(
            index=["minimize", "ideal", "nadir"],
            columns=self._problem.get_objective_names(),
        )
        dimensions_data.loc["minimize"] = self._problem._max_multiplier
        dimensions_data.loc["ideal"] = self._ideal
        dimensions_data.loc["nadir"] = self._nadir

        data = pd.DataFrame(
            objectives, columns=self._problem.get_objective_names()
        )

        plot_request = SimplePlotRequest(
            data=data,
            dimensions_data=dimensions_data,
            message="Computed new solutions",
        )

        return save_request, plot_request

    def iterate(
        self,
        request: Union[
            NimbusClassificationRequest,
            NimbusSaveRequest,
            NimbusIntermediateSolutionsRequest,
        ],
    ) -> Tuple[
        Union[
            NimbusClassificationRequest,
            NimbusSaveRequest,
            NimbusIntermediateSolutionsRequest,
        ],
        Union[SimplePlotRequest, None],
    ]:
        if self._state == "classify":
            if type(request) != NimbusClassificationRequest:
                print("ERROR")
            else:
                try:
                    requests = self.handle_classification_request(request)
                except Exception as _:
                    # handle me
                    pass

                # succesfully generated new requests, change state
                self._state = "archive"
                return requests

        if self._state == "archive":
            if type(request) != NimbusSaveRequest:
                print("ERROR")
            else:
                self.handle_save_request(request)


if __name__ == "__main__":
    from desdeo_problem.Problem import MOProblem
    from desdeo_problem.Objective import _ScalarObjective
    from desdeo_problem.Variable import variable_builder
    from desdeo_problem.Constraint import ScalarConstraint
    from desdeo_tools.scalarization.Scalarizer import Scalarizer

    # create the problem
    def f_1(x):
        res = 4.07 + 2.27 * x[:, 0]
        return -res

    def f_2(x):
        res = (
            2.60
            + 0.03 * x[:, 0]
            + 0.02 * x[:, 1]
            + 0.01 / (1.39 - x[:, 0] ** 2)
            + 0.30 / (1.39 - x[:, 1] ** 2)
        )
        return -res

    def f_3(x):
        res = 8.21 - 0.71 / (1.09 - x[:, 0] ** 2)
        return -res

    def f_4(x):
        res = 0.96 - 0.96 / (1.09 - x[:, 1] ** 2)
        return -res

    def f_5(x):
        return np.max([np.abs(x[:, 0] - 0.65), np.abs(x[:, 1] - 0.65)], axis=0)

    def c_1(x, f=None):
        x = x.squeeze()
        return (x[0] + x[1]) - 0.5

    f1 = _ScalarObjective(name="f1", evaluator=f_1)
    f2 = _ScalarObjective(name="f2", evaluator=f_2)
    f3 = _ScalarObjective(name="f3", evaluator=f_3)
    f4 = _ScalarObjective(name="f4", evaluator=f_4)
    f5 = _ScalarObjective(name="f5", evaluator=f_5)
    varsl = variable_builder(
        ["x_1", "x_2"],
        initial_values=[0.5, 0.5],
        lower_bounds=[0.3, 0.3],
        upper_bounds=[1.0, 1.0],
    )
    c1 = ScalarConstraint("c1", 2, 5, evaluator=c_1)
    problem = MOProblem(
        variables=varsl, objectives=[f1, f2, f3, f4, f5], constraints=[c1]
    )

    method = NIMBUS(problem)
    reqs = method.requests()[0]
    response = {}
    response["classifications"] = ["<", "<=", "=", ">=", "0"]
    response["levels"] = [-6, -3, -5, 8, 0.349]
    response["number_of_solutions"] = 4
    reqs._response = response
    res = method.iterate(reqs)[0]

    print(method._archive_objectives)
    print(method._archive_solutions)
    res._response = {"indices": [0, 2]}
    method.iterate(res)
    print(method._archive_objectives)
    print(method._archive_solutions)
