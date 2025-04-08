import datetime
from models.marko_qp.slsqp import slsqp
from models.marko_qp.qp import qp
from models.marko_cpsat.markowitz_cpsat import cpsat
from models.marko_ga.ga import ga
from models.cvar_minlp.cvar_minlp import cvar_minlp

def select_model(
    model: str,
    tickers: list,
    start_date: datetime,
    end_date: datetime,
    bounds : list,
    risk : int
):
    match model:
        case "QP":
            return qp(tickers, start_date, end_date, bounds, None)
        case "SLSQP":
            return slsqp(tickers, start_date, end_date, bounds, risk)
        case "GA(Arithmetic-Gaussian-Tournament-Dirichlet)":
            return ga("agtd.yaml")(tickers, start_date, end_date, bounds, risk)
        case "GA(Convex-Gaussian-Tournament-Dirichlet)":
            return ga("cgtd.yaml")(tickers, start_date, end_date, bounds, risk)
        case "GA(Convex-Directional-Tournament-Dirichlet)":
            return ga("cdtd.yaml")(tickers, start_date, end_date, bounds, risk)
        case "GA(Convex-Gaussian-Best-Dirichlet)":
            return ga("cgbd.yaml")(tickers, start_date, end_date, bounds, risk)
        case "GA(Convex-Gaussian-Best-Uniform)":
            return ga("cgbu.yaml")(tickers, start_date, end_date, bounds, risk)
        case "CP_SAT":
            return cpsat(tickers, start_date, end_date, bounds, risk)
        case "MINLP":
            return cvar_minlp(tickers, start_date, end_date, bounds, risk)
        case _:
            print(f"ERROR - model {model} should not exists")
            return {"weights": []}
    return []
