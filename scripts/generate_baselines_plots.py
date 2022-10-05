import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from typing import List
import re
from loguru import logger
from collections import OrderedDict

import socket


def get_ylim(county_name):
    county_ylims = [
        (".*miamidade.*", 250000),  # int(2716940 * 0.7)),
        (".*losangeles.*", 400000),  # int(10039107 * 0.7)),
        (".*middlesex.*", 50000),  # int(1611699 * 0.7)),
    ]
    default_ylim = 250000
    for regex, ylim in county_ylims:
        if re.search(regex, county_name) is not None:
            return ylim
    return default_ylim


def get_ylim_deaths(county_name):
    county_ylims = [
        (".*miamidade.*", 5000),  # int(2716940 * 0.7)),
        (".*losangeles.*", 8000),  # int(10039107 * 0.7)),
        (".*middlesex.*", 3000),  # int(1611699 * 0.7)),
    ]
    default_ylim = 250000
    for regex, ylim in county_ylims:
        if re.search(regex, county_name) is not None:
            return ylim
    return default_ylim


def compute_mdae(tensor1, tensor2, county_pop):
    return ((tensor1 - tensor2).abs().mean() / county_pop).item()


def load_csv(csv, county, state, begin=11 + 38, end=11 + 200):
    df = pd.read_csv(csv)
    return query_df(df, county, state, begin, end)


def query_df(df, county, state, begin=11 + 38, end=11 + 200):
    a = df.query(f'Admin2 == "{county}" & Province_State == "{state}"').to_numpy()[:, begin:end]
    return torch.tensor(a.astype(np.float32))


def plot_generated_trajectories(ground_truth, generated, basename, county_pop, ylim, ylabel, title=None):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(generated[0], c="C0", alpha=0.1, label="samples")
    ax.plot(generated[1:].T, c="C0", alpha=0.1)

    quantile = generated.quantile(torch.tensor([0.25, 0.5, 0.75]), dim=0)
    ax.plot(quantile[0], "--", c="k", alpha=0.5, label="quartiles")
    ax.plot(quantile[1], c="k", label="median")
    ax.plot(quantile[2], "--", c="k", alpha=0.5)

    ax.plot(ground_truth.T, c="C1", linewidth=3, label="ground truth")
    ax.set_xlim(0, 160)
    ax.set_ylim(0, ylim)
    ax.legend()
    ax.set_xlabel("Simulation Day")
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    mdae = compute_mdae(generated, ground_truth, county_pop)

    fig.savefig(basename + ".png", bbox_inches="tight")
    np.savetxt(basename + "_mdae.txt", [mdae])
    plt.close()
    return mdae


def get_param_summary(log):
    with open(log, "r") as f:
        lines = f.read().splitlines()

    arg_lines = []
    final_section = None
    for i, line in enumerate(lines):
        if line.startswith("Final"):
            final_section = lines[i + 1 :]
        if line.startswith("Dict{Symbol,Any}"):
            arg_lines.append(line)

    # Dict{Symbol,Any}(:path_out => "/scratch/rwalters/ProbProgEpiNet.jl/src/../output/exponential_transition_6",:tiny => false,:io => IOStream(<file /scratch/rwalters/ProbProgEpiNet.jl/src/../output/exponential_transition_6/time=2021-04-20T17:31:19,git_hash=1df42da+,noise=0.005,lr=5.0e-6,lead_in=7,inf_thresh=0.01,duration=163,inf_sat=0.5,iters=30,fixed_E0=0.01,gam_lam_std=-2.3,seed=1,county=losangeles-exp,noise_fn=mag_scaling,sc=5.0e-5,/log.txt>),:end_day => 225,:output_dir_base => "output/exponential_transition_6",:mode => "infer",:jhu_county_name => "Los Angeles",:real_deaths => "/scratch/rwalters/ProbProgEpiNet.jl/src/../COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv",:samples_per_iter => 120,:final_inf_saturation_frac => 0.5,:lr => 5.0e-6,:noise_fn => "mag_scaling",:jhu_state_name => "California",:decay => 10,:obs_noise_std => 0.005,:multithreaded => false,:json_results => nothing,:county_pop => 10039107,:git_hash => "1df42da+",:path_data => "/scratch/rwalters/ProbProgEpiNet.jl/src/../COVID-19/csse_covid_19_data/csse_covid_19_time_series",:county_mortality_rate => 0.0231,:mortality_rate => 0.009515929653902484,:output_dir => "/scratch/rwalters/ProbProgEpiNet.jl/src/../output/exponential_transition_6/time=2021-04-20T17:31:19,git_hash=1df42da+,noise=0.005,lr=5.0e-6,lead_in=7,inf_thresh=0.01,duration=163,inf_sat=0.5,iters=30,fixed_E0=0.01,gam_lam_std=-2.3,seed=1,county=losangeles-exp,noise_fn=mag_scaling,sc=5.0e-5,",:iterations => 30,:inf_thresh_after_lead => 0.01,:scaling_factor => 5.0e-5,:lead_in_time => 7,:prior_lambda_logit_mean => -2.56,:path_node_attributes => "/scratch/rwalters/ProbProgEpiNet.jl/src/../ExperimentData/updated_network_model/losangeles/20_cbgs/01/node_attributes.json",:output_terms => true,:json_results_name => "posterior_params.json",:num_traj => 100,:path_edge_attributes => "/scratch/rwalters/ProbProgEpiNet.jl/src/../ExperimentData/updated_network_model/losangeles/20_cbgs/01/edge_attributes_02-17.json",:county => "losangeles-exp",:prior_gamma_logit_mean => -1.79,:real_confirmed => "/scratch/rwalters/ProbProgEpiNet.jl/src/../COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv",:start_day => 62,:prior_E0_logit_std_L => 0.0,:duration => 163,:knots => 6,:timestamp => "2021-04-20T17:31:19",:fixed_total_E0 => 0.01,:cond_on_deaths => false,:prior_gamma_lambda_logit_std_L => -2.3,:validation_type => nothing,:death_undercount_factor => 10,:random_seed => 1,:prior_betaE_logit_mean => -2.64,:prior_beta_logit_std_L => -2.0,:experiment_config => "./config.json",:prior_E0_logit_mean => -7.0,:undercount_factor => 24.27508487363259,:json_results_checkpoint_name => "checkpoint_posterior_params.json")

    arg_line = arg_lines[1]
    split_args_raw = re.split("=>|,:", arg_line[18:-1])
    split_args = [str.strip(s) for s in split_args_raw]

    arg_dict = OrderedDict()
    for i in range(0, len(split_args), 2):
        arg_dict[split_args[i]] = split_args[i + 1]
    arg_dict.pop("io")

    if final_section is not None:
        # Get betas
        for i, line in enumerate(final_section):
            if line.startswith("βE summary"):
                beta_e_summary = final_section[i + 1 : i + 1 + 4]

        prefix = "Knot means: Any["
        suffix = "]"
        knots = beta_e_summary[0][len(prefix) : -len(suffix)].replace(" ", "").split(",")

        prefix = "Average: "
        avg_beta_e = beta_e_summary[1][len(prefix) :]

        prefix = "Average first 32 days: "
        avg_first_32 = beta_e_summary[2][len(prefix) :]

        prefix = "Highest 32 day average: "
        highest_avg = beta_e_summary[3][len(prefix) :]

        # Get gamma
        for i, line in enumerate(final_section):
            if line.startswith("γ"):
                gamma_summary = final_section[i + 1 : i + 1 + 8]

        prefix = '("post med        ", ['
        suffix = "])"
        # gamma_post_median = gamma_summary[5][len(prefix) : -len(suffix)]
        gamma_post_median = None

        prefix = '("prior med       ", '
        suffix = ")"
        # gamma_prior_median = gamma_summary[1][len(prefix) : -len(suffix)]
        gamma_prior_median = None

        # Get lambda
        for i, line in enumerate(final_section):
            if line.startswith("λ"):
                lambda_summary = final_section[i + 1 : i + 1 + 8]

        prefix = '("post med        ", ['
        suffix = "])"
        # lambda_post_median = lambda_summary[5][len(prefix) : -len(suffix)]
        lambda_post_median = None

        prefix = '("prior med       ", '
        suffix = ")"
        # lambda_prior_median = lambda_summary[1][len(prefix) : -len(suffix)]
        lambda_prior_median = None

        return (
            knots,
            [avg_beta_e, avg_first_32, highest_avg],
            [gamma_prior_median, gamma_post_median],
            [lambda_prior_median, lambda_post_median],
            arg_dict,
        )
    else:
        return ([None] * 6, [None, None, None], [None, None], [None, None], arg_dict)


def parse_filename(county_name: str, is_size_varying=True):
    # Example:
    # git_hash=15bd75c,time=2021-03-12T13:43:52,noise=0.0005,lr=5.0e-6,decay=10,s_per_iter=120,iters=20,E0_mean=-7.0,E0_std=0.0,betaE_mean=-1.39,dis_param_std=0.0,seed=1,SEK=38_200_6,county=losangeles__02-03__20_cbgs__03
    if not is_size_varying:
        return county_name, "Unknown", "Unknown", "Unknown"
    info = county_name.split("county=")[1]
    county, date, size, seed = info.split("__")
    size = size[:-5]
    return county, date, size, seed


def get_one_field(name, field, sep="=", delim=","):
    # fields delimited by "," - assumes no comma in county name!
    # 'git_hash=4be6ca0,time=2021-03-29T19:16:07,noise=0.0005,lr=5.0e-6,iters=200,seed=1,county=losangeles-exp,CondOnDnoise_fn=day_scaling,sc=0.0005,'
    # fields delimited by "," - assumes no comma in county name!
    match = re.search(f"{field}{sep}(.*?){delim}", name)
    if match is None:
        return None
    return match.group(1)


def get_csv_from_results(script_path, folders: List[Path], output_path, county_conf, is_size_varying=False):
    jhu_df = pd.read_csv(
        script_path
        / ".."
        / "COVID-19"
        / "csse_covid_19_data"
        / "csse_covid_19_time_series"
        / "time_series_covid19_confirmed_US.csv"
    )
    csv_rows = []

    # NOTE - hardcoded knots=6
    knots_header = [f"betaE_knot{i}" for i in range(6)]
    csv_header_list = [
        "county",
        "date",
        "size",
        "graph_seed",
        "our_seed",
        "iters",
        "noise",
        "noise_fn",
        "scaling_factor",
        "cond_on_d",
        "mdae",
        *knots_header,
        "avg_betaE",
        "avg_betaE_first32",
        "avg_betaE_highest32",
        "gamma_prior",
        "gamma_post",
        "lambda_prior",
        "lambda_post",
        "hostname",
        "inf_zero_var",
        "E0_medians",
        "seir_zero_var",
        "custom_inf_plot",
    ]

    hostname = socket.gethostname()

    arg_dict_keys = []
    for f in tqdm(folders, desc="Folders", leave=True):
        county_name = get_one_field(f.name, "county")
        if county_name is None:
            logger.warning(f"County name not matched: {f}")
            continue
        csv_row = []

        ylim = get_ylim(county_name)

        our_seed = get_one_field(f.name, "seed")
        iters = get_one_field(f.name, "iters")
        noise = get_one_field(f.name, "noise")
        noise_fn = get_one_field(f.name, "noise_fn")
        scaling_factor = get_one_field(f.name, "sc")
        cond_on_d = get_one_field(f.name, "CondOnD", sep="", delim="") is not None

        jhu_county = county_conf[county_name]["jhu_county_name"]
        jhu_state = county_conf[county_name]["jhu_state_name"]
        county_pop = county_conf[county_name]["county_pop"]

        # If this is a size-varying expt, we'll further split county name.
        # Otherwise, county_name will be unchanged and the rest will be "Unknown"
        county_name, date, size, graph_seed = parse_filename(county_name, is_size_varying)

        try:
            knots, betas, gammas, lambdas, arg_dict = get_param_summary(f / "log.txt")
            arg_dict_keys = arg_dict.keys()

        except:
            continue

        # Get original row from JHU
        original = query_df(
            jhu_df, jhu_county, jhu_state, int(arg_dict["start_day"]) + 11 + 1, int(arg_dict["end_day"]) + 11
        )

        # Done getting experiment details
        csv_row.extend(
            [
                county_name,
                date,
                size,
                graph_seed,
                our_seed,
                iters,
                noise,
                noise_fn,
                scaling_factor,
                cond_on_d,
            ]
        )

        try:
            knots, betas, gammas, lambdas, arg_dict = get_param_summary(f / "log.txt")
            arg_dict_keys = arg_dict.keys()
        except:
            continue

        # Get generated row from model
        generated_traj_file = f / "generated_trajectories_confirmed.csv"
        if not generated_traj_file.exists():
            # Now we know that run failed
            # missing_fields = len(csv_header_list) - len(csv_row)
            # csv_row.extend([None] * missing_fields)
            csv_row.extend(
                [
                    None,
                    *knots,
                    *betas,
                    *gammas,
                    *lambdas,
                    None,
                    None,
                    None,
                    None,
                    None,
                    *arg_dict.values(),
                ]
            )
            csv_rows.append(",".join(map(str, csv_row)))

            continue

        generated = load_csv(
            generated_traj_file,
            jhu_county,
            jhu_state,
            int(arg_dict["start_day"]) + 11 + 1,
            int(arg_dict["end_day"]) + 11,
        )
        basename = str(f / county_name)

        img_paths = [
            f'"{str(f / "cumulative_inf_comparison.zero_var.num_traj=100.png")}"',
            f'"{str(f / "E0_medians.png")}"',
            f'"{str(f / "posterior_seir_curve.zero_var.num_traj=100.png")}"',
            # The image created by plot_generated_trajectories()
            f'"{str(f / f"{basename}.png")}"',
        ]

        mdae = plot_generated_trajectories(original, generated, basename, county_pop, ylim, "Cumulative Infected")

        csv_row.extend(
            [
                str(mdae),
                *knots,
                *betas,
                *gammas,
                *lambdas,
                hostname,
                *img_paths,
                *arg_dict.values(),
            ]
        )

        csv_rows.append(",".join(map(str, csv_row)))

    # TODO - maybe csv writer?

    csv_header_list.extend(arg_dict_keys)
    csv_header = ",".join(csv_header_list)

    output_path.mkdir(exist_ok=True)
    with open(output_path / "results.csv", "w") as f:
        f.write(csv_header + "\n")
        for line in csv_rows:
            f.write(line + "\n")


def get_csv_from_results_deaths(script_path, folders: List[Path], output_path, county_conf, is_size_varying=False):
    jhu_df = pd.read_csv(
        script_path
        / ".."
        / "COVID-19"
        / "csse_covid_19_data"
        / "csse_covid_19_time_series"
        / "time_series_covid19_deaths_US.csv"
    )
    csv_rows = []

    # NOTE - hardcoded knots=6
    knots_header = [f"betaE_knot{i}" for i in range(6)]
    csv_header_list = [
        "county",
        "date",
        "size",
        "graph_seed",
        "our_seed",
        "iters",
        "noise",
        "noise_fn",
        "scaling_factor",
        "cond_on_d",
        "mdae",
        *knots_header,
        "avg_betaE",
        "avg_betaE_first32",
        "avg_betaE_highest32",
        "gamma_prior",
        "gamma_post",
        "lambda_prior",
        "lambda_post",
        "hostname",
        "inf_zero_var",
        "E0_medians",
        "seir_zero_var",
        "custom_inf_plot",
    ]

    hostname = socket.gethostname()

    arg_dict_keys = []
    for f in tqdm(folders, desc="Folders", leave=True):
        county_name = get_one_field(f.name, "county")
        if county_name is None:
            logger.warning(f"County name not matched: {f}")
            continue
        csv_row = []

        ylim = get_ylim_deaths(county_name)

        our_seed = get_one_field(f.name, "seed")
        iters = get_one_field(f.name, "iters")
        noise = get_one_field(f.name, "noise")
        noise_fn = get_one_field(f.name, "noise_fn")
        scaling_factor = get_one_field(f.name, "sc")
        cond_on_d = get_one_field(f.name, "CondOnD", sep="", delim="") is not None

        jhu_county = county_conf[county_name]["jhu_county_name"]
        jhu_state = county_conf[county_name]["jhu_state_name"]
        county_pop = county_conf[county_name]["county_pop"]

        # If this is a size-varying expt, we'll further split county name.
        # Otherwise, county_name will be unchanged and the rest will be "Unknown"
        county_name, date, size, graph_seed = parse_filename(county_name, is_size_varying)

        try:
            knots, betas, gammas, lambdas, arg_dict = get_param_summary(f / "log.txt")
            arg_dict_keys = arg_dict.keys()

        except:
            continue

        # Get original row from JHU
        # NOTE - deaths CSV has an extra column for "population" - skip one more col
        original = query_df(
            jhu_df, jhu_county, jhu_state, int(arg_dict["start_day"]) + 11 + 1 + 1, int(arg_dict["end_day"]) + 11
        )

        # Done getting experiment details
        csv_row.extend(
            [
                county_name,
                date,
                size,
                graph_seed,
                our_seed,
                iters,
                noise,
                noise_fn,
                scaling_factor,
                cond_on_d,
            ]
        )

        try:
            knots, betas, gammas, lambdas, arg_dict = get_param_summary(f / "log.txt")
            arg_dict_keys = arg_dict.keys()
        except:
            continue

        # Get generated row from model
        generated_traj_file = f / "generated_trajectories_deaths.csv"
        if not generated_traj_file.exists():
            # Now we know that run failed
            # missing_fields = len(csv_header_list) - len(csv_row)
            # csv_row.extend([None] * missing_fields)
            csv_row.extend(
                [
                    None,
                    *knots,
                    *betas,
                    *gammas,
                    *lambdas,
                    None,
                    None,
                    None,
                    None,
                    None,
                    *arg_dict.values(),
                ]
            )
            csv_rows.append(",".join(map(str, csv_row)))

            continue

        # NOTE - again skip 1 more column; we mimic the header from the JHU data file
        generated = load_csv(
            generated_traj_file,
            jhu_county,
            jhu_state,
            int(arg_dict["start_day"]) + 11 + 1 + 1,
            int(arg_dict["end_day"]) + 11,
        )
        basename = str(f / county_name) + "_deaths"

        img_paths = [None, None, None, None]

        mdae = plot_generated_trajectories(original, generated, basename, county_pop, ylim, "Cumulative Deaths")

        csv_row.extend(
            [
                str(mdae),
                *knots,
                *betas,
                *gammas,
                *lambdas,
                hostname,
                *img_paths,
                *arg_dict.values(),
            ]
        )

        csv_rows.append(",".join(map(str, csv_row)))

    # TODO - maybe csv writer?

    csv_header_list.extend(arg_dict_keys)
    csv_header = ",".join(csv_header_list)

    output_path.mkdir(exist_ok=True)
    with open(output_path / "results_deaths.csv", "w") as f:
        f.write(csv_header + "\n")
        for line in csv_rows:
            f.write(line + "\n")


def baseline_mdae_comparison(script_path, county_conf, county_ylims):
    data_root = script_path / ".." / "baseline_comparison"

    models = ["analytic"]
    e0_fracs = ["0.005", "0.01", "0.02", "0.04", "ours-tight", "ours-loose"]

    jhu_df = pd.read_csv(
        script_path
        / ".."
        / "COVID-19"
        / "csse_covid_19_data"
        / "csse_covid_19_time_series"
        / "time_series_covid19_confirmed_US.csv"
    )

    title = None

    for c, ylim in county_ylims.items():
        for m in models:
            for e in e0_fracs:
                county = county_conf[c]["jhu_county_name"]
                state = county_conf[c]["jhu_state_name"]
                county_pop = county_conf[c]["county_pop"]

                # Get original row from JHU
                original = query_df(jhu_df, county, state)

                # Get generated row from model
                folder = data_root / c / m / e
                generated = load_csv(folder / "generated_trajectories_confirmed.csv", county, state)
                basename = str(folder / f"baseline_params__{c}__{m}__{e}")

                plot_generated_trajectories(original, generated, basename, county_pop, ylim, title)


def cycle_validation(script_path, county_conf, county_ylims):
    data_root = script_path / ".." / "validation_cycle" / "vary_beta"

    scenarios = ["hi", "hi_lo", "hi_lo_hi", "lo", "lo_hi", "lo_hi_lo"]

    title = None

    for c, ylim in county_ylims.items():
        for s in scenarios:
            folder = data_root / c / s
            county_pop = county_conf[c]["county_pop"]
            county = county_conf[c]["jhu_county_name"]
            state = county_conf[c]["jhu_state_name"]

            ground_truth = load_csv(folder / "generated_trajectories_confirmed.csv", county, state)
            generated = load_csv(
                folder / "validation_results" / "generated_trajectories_confirmed.csv",
                county,
                state,
            )
            basename = str(folder / f"cycle_validation__{c}__{s}")
            plot_generated_trajectories(ground_truth, generated, basename, county_pop, ylim, title)


def our_mdae_comparison(script_path, county_conf, county_ylims):
    data_root = script_path / ".." / "baseline_comparison" / "data"

    scenarios = {
        "Nov-miamidade": ["miami-tight", "miami-loose"],
        "Nov-losangeles": ["la-tight", "la-loose"],
    }

    jhu_df = pd.read_csv(
        script_path
        / ".."
        / "COVID-19"
        / "csse_covid_19_data"
        / "csse_covid_19_time_series"
        / "time_series_covid19_confirmed_US.csv"
    )

    title = None

    for c, ylim in county_ylims.items():
        for s in scenarios[c]:
            county_pop = county_conf[c]["county_pop"]
            county = county_conf[c]["jhu_county_name"]
            state = county_conf[c]["jhu_state_name"]

            # Original row from JHU
            original = query_df(jhu_df, county, state)

            # Our generated rows
            folder = data_root
            generated = load_csv(folder / (s + ".csv"), county, state)
            basename = str(folder / s)

            plot_generated_trajectories(original, generated, basename, county_pop, ylim, title)


if __name__ == "__main__":
    import argparse

    script_path = Path(__file__).parent.absolute()

    p = argparse.ArgumentParser()
    p.add_argument(
        "--data-path",
        type=Path,
        default=Path("/data/shared/coanet_results/output_4_15_21/exponential_transition_4_15"),
    )
    p.add_argument(
        "--config",
        type=Path,
        default=Path(script_path / ".." / "config.json"),
    )
    args = p.parse_args()

    with open(args.config) as f:
        county_conf = json.load(f)

    get_csv_from_results(
        script_path,
        folders=list(args.data_path.iterdir()),
        output_path=args.data_path / "results",
        county_conf=county_conf,
        is_size_varying=False,
    )
    get_csv_from_results_deaths(
        script_path,
        folders=list(args.data_path.iterdir()),
        output_path=args.data_path / "results",
        county_conf=county_conf,
        is_size_varying=False,
    )
    # cycle_validation(script_path, county_conf, county_ylims)
    # baseline_mdae_comparison(script_path, county_conf, county_ylims)
    # our_mdae_comparison(script_path, county_conf, county_ylims)
