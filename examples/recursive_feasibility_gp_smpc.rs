use grampc_s_rs::{GaussianProcess, GaussianProcessData, Result, SquaredExponentialKernel};
use nalgebra::{DMatrix, DVector};

const HORIZON: usize = 4;
const DATA_POINTS: usize = 10;
const PENALTY: f64 = 1_000_000.0;

type ScalarGp = GaussianProcess<SquaredExponentialKernel>;

#[derive(Debug, Clone)]
struct PaperProblem {
    gp_x1: ScalarGp,
    gp_v: ScalarGp,
}

#[derive(Debug, Clone)]
struct Tightening {
    c: Vec<f64>,
    delta: Vec<f64>,
    sigma_max: f64,
    b_max: f64,
    l_f: f64,
    l_g: f64,
    l_mu: f64,
}

#[derive(Debug, Clone)]
struct Solution {
    controls: [f64; HORIZON],
    cost: f64,
    max_violation: f64,
    mean_trajectory: Vec<DVector<f64>>,
    true_trajectory: Vec<DVector<f64>>,
}

fn main() -> Result<()> {
    let problem = PaperProblem::new()?;
    let tightening = problem.tightening()?;

    println!("paper: examples/papers/recursive_feasibility_nonlinear_stochastic_mpc_gp.pdf");
    println!("section 5 nonlinear GP-SMPC numerical evaluation");
    println!("horizon: {HORIZON}");
    println!(
        "bounds: sigma_max={:.6}, b_max={:.6}, Lf={:.6}, Lg={:.6}, Lmu={:.6}",
        tightening.sigma_max, tightening.b_max, tightening.l_f, tightening.l_g, tightening.l_mu
    );
    println!("c_i:     {:?}", rounded(&tightening.c));
    println!("delta_i: {:?}", rounded(&tightening.delta));

    for x0 in [
        DVector::from_vec(vec![3.0, -3.0]),
        DVector::from_vec(vec![-3.0, 1.0]),
    ] {
        let solution = solve_initial_state(&problem, &tightening, &x0)?;
        println!();
        println!("initial state: {}", x0.transpose());
        println!("optimized tightened cost: {:.6}", solution.cost);
        println!("max tightened violation: {:.6}", solution.max_violation);
        println!("controls: {:?}", rounded_array(&solution.controls));
        println!("predicted mean trajectory:");
        for (i, x) in solution.mean_trajectory.iter().enumerate() {
            println!("  i={i}: {}", x.transpose());
        }
        println!("true trajectory under the same controls:");
        for (i, x) in solution.true_trajectory.iter().enumerate() {
            println!("  i={i}: {}", x.transpose());
        }
        let variances =
            problem.gp_variance_trace_along(&solution.mean_trajectory, &solution.controls)?;
        println!(
            "GP variance trace along prediction: {:?}",
            rounded(&variances)
        );
    }

    Ok(())
}

impl PaperProblem {
    fn new() -> Result<Self> {
        Ok(Self {
            gp_x1: make_gp(
                -3.0,
                3.0,
                |x| 0.1 * x.tanh(),
                vec![true, false],
                vec![false],
            )?,
            gp_v: make_gp(-2.0, 2.0, |v| 0.1 * v.sin(), vec![false, false], vec![true])?,
        })
    }

    fn known_next(&self, x: &DVector<f64>, v: f64) -> DVector<f64> {
        DVector::from_vec(vec![x[0] + 0.5 * (x[1] + v), x[1] + v])
    }

    fn true_next(&self, x: &DVector<f64>, v: f64) -> DVector<f64> {
        self.known_next(x, v) + DVector::from_vec(vec![0.1 * x[0].tanh(), 0.1 * v.sin()])
    }

    fn model_next(&self, x: &DVector<f64>, v: f64) -> Result<DVector<f64>> {
        let u = DVector::from_vec(vec![v]);
        Ok(self.known_next(x, v)
            + DVector::from_vec(vec![self.gp_x1.mean(x, &u)?, self.gp_v.mean(x, &u)?]))
    }

    fn gp_variance_trace(&self, x: &DVector<f64>, v: f64) -> Result<f64> {
        let u = DVector::from_vec(vec![v]);
        Ok(self.gp_x1.variance(x, &u)?.max(0.0) + self.gp_v.variance(x, &u)?.max(0.0))
    }

    fn gp_variance_trace_along(
        &self,
        states: &[DVector<f64>],
        controls: &[f64; HORIZON],
    ) -> Result<Vec<f64>> {
        let mut out = Vec::with_capacity(HORIZON);
        for i in 0..HORIZON {
            out.push(self.gp_variance_trace(&states[i], controls[i])?);
        }
        Ok(out)
    }

    fn tightening(&self) -> Result<Tightening> {
        let sigma_max = self.sigma_max()?;
        let b_max = 1.5 * rkhs_norm_bound();
        let l_f = known_dynamics_lipschitz();
        let l_g: f64 = 0.1;
        let l_mu = self.gp_mean_lipschitz_estimate()?;

        let mut c: Vec<f64> = vec![0.0; HORIZON + 1];
        let mut delta: Vec<f64> = vec![0.0; HORIZON + 1];
        for i in 0..HORIZON {
            c[i + 1] = ((l_f + l_mu).powi(2) * c[i].powi(2) + sigma_max.powi(2)).sqrt();
            delta[i + 1] = (l_f + l_g.max(l_mu)) * (delta[i] + c[i]) + b_max * sigma_max + delta[i];
        }

        Ok(Tightening {
            c,
            delta,
            sigma_max,
            b_max,
            l_f,
            l_g,
            l_mu,
        })
    }

    fn sigma_max(&self) -> Result<f64> {
        let mut max_x = 0.0_f64;
        let mut max_v = 0.0_f64;
        for i in 0..=400 {
            let x1 = -3.0 + 6.0 * (i as f64) / 400.0;
            let state = DVector::from_vec(vec![x1, 0.0]);
            let zero_u = DVector::from_vec(vec![0.0]);
            max_x = max_x.max(self.gp_x1.variance(&state, &zero_u)?.max(0.0));

            let v = -2.0 + 4.0 * (i as f64) / 400.0;
            let zero_state = DVector::from_vec(vec![0.0, 0.0]);
            let control = DVector::from_vec(vec![v]);
            max_v = max_v.max(self.gp_v.variance(&zero_state, &control)?.max(0.0));
        }
        Ok((max_x + max_v).sqrt())
    }

    fn gp_mean_lipschitz_estimate(&self) -> Result<f64> {
        let mut max_grad = 0.0_f64;
        let control = DVector::from_vec(vec![0.0]);
        for i in 0..=400 {
            let x1 = -3.0 + 6.0 * (i as f64) / 400.0;
            let state = DVector::from_vec(vec![x1, 0.0]);
            max_grad = max_grad.max(self.gp_x1.mean_gradient_state(&state, &control)?.norm());
        }
        Ok(max_grad)
    }
}

fn solve_initial_state(
    problem: &PaperProblem,
    tightening: &Tightening,
    x0: &DVector<f64>,
) -> Result<Solution> {
    let candidates = [
        [0.0, 0.0, 0.0, 0.0],
        [0.6, 0.7, 0.6, 0.5],
        [1.0, -0.3, -0.55, -0.65],
        [-0.5, -0.5, -0.5, -0.5],
        [1.5, 0.0, -0.5, -0.5],
    ];

    let mut best = None;
    for candidate in candidates {
        let controls = optimize_from(problem, tightening, x0, candidate)?;
        let (cost, max_violation, mean_trajectory) = objective(problem, tightening, x0, &controls)?;
        let true_trajectory = true_rollout(problem, x0, &controls);
        let solution = Solution {
            controls,
            cost,
            max_violation,
            mean_trajectory,
            true_trajectory,
        };
        if best
            .as_ref()
            .map(|current: &Solution| solution.cost < current.cost)
            .unwrap_or(true)
        {
            best = Some(solution);
        }
    }

    Ok(best.expect("at least one candidate"))
}

fn optimize_from(
    problem: &PaperProblem,
    tightening: &Tightening,
    x0: &DVector<f64>,
    mut controls: [f64; HORIZON],
) -> Result<[f64; HORIZON]> {
    let mut best = objective(problem, tightening, x0, &controls)?.0;
    for _ in 0..600 {
        let gradient = finite_difference_gradient(problem, tightening, x0, &controls)?;
        let grad_norm = gradient
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
        if grad_norm < 1e-7 {
            break;
        }

        let mut accepted = false;
        let mut step = 0.05;
        while step > 1e-12 {
            let mut candidate = controls;
            for i in 0..HORIZON {
                candidate[i] = (candidate[i] - step * gradient[i]).clamp(-2.0, 2.0);
            }
            let cost = objective(problem, tightening, x0, &candidate)?.0;
            if cost < best {
                controls = candidate;
                best = cost;
                accepted = true;
                break;
            }
            step *= 0.5;
        }

        if !accepted {
            break;
        }
    }
    Ok(controls)
}

fn finite_difference_gradient(
    problem: &PaperProblem,
    tightening: &Tightening,
    x0: &DVector<f64>,
    controls: &[f64; HORIZON],
) -> Result<[f64; HORIZON]> {
    let mut gradient = [0.0; HORIZON];
    let eps = 1e-5;
    for i in 0..HORIZON {
        let mut plus = *controls;
        let mut minus = *controls;
        plus[i] += eps;
        minus[i] -= eps;
        gradient[i] = (objective(problem, tightening, x0, &plus)?.0
            - objective(problem, tightening, x0, &minus)?.0)
            / (2.0 * eps);
    }
    Ok(gradient)
}

fn objective(
    problem: &PaperProblem,
    tightening: &Tightening,
    x0: &DVector<f64>,
    controls: &[f64; HORIZON],
) -> Result<(f64, f64, Vec<DVector<f64>>)> {
    let mut x = x0.clone();
    let mut cost = 0.0;
    let mut max_violation = 0.0_f64;
    let mut trajectory = Vec::with_capacity(HORIZON + 1);
    trajectory.push(x.clone());

    for i in 0..HORIZON {
        let v = controls[i];
        cost += x.dot(&x) + v.powi(2);
        for h in [
            x[0].abs() - 3.0 + tightening.delta[i],
            x[1].abs() - 3.0 + tightening.delta[i],
            v.abs() - 2.0,
        ] {
            if h > 0.0 {
                cost += PENALTY * h.powi(2);
                max_violation = max_violation.max(h);
            }
        }
        x = problem.model_next(&x, v)?;
        trajectory.push(x.clone());
    }

    cost += 2.0 * x.dot(&x);
    for h in [
        x[0].abs() - 1.0 + tightening.delta[HORIZON],
        x[1].abs() - 1.0 + tightening.delta[HORIZON],
    ] {
        if h > 0.0 {
            cost += PENALTY * h.powi(2);
            max_violation = max_violation.max(h);
        }
    }

    Ok((cost, max_violation, trajectory))
}

fn true_rollout(
    problem: &PaperProblem,
    x0: &DVector<f64>,
    controls: &[f64; HORIZON],
) -> Vec<DVector<f64>> {
    let mut x = x0.clone();
    let mut trajectory = Vec::with_capacity(HORIZON + 1);
    trajectory.push(x.clone());
    for &v in controls {
        x = problem.true_next(&x, v);
        trajectory.push(x.clone());
    }
    trajectory
}

fn make_gp(
    lower: f64,
    upper: f64,
    output: impl Fn(f64) -> f64,
    state_dependency: Vec<bool>,
    control_dependency: Vec<bool>,
) -> Result<ScalarGp> {
    let mut inputs = Vec::with_capacity(DATA_POINTS);
    let mut outputs = Vec::with_capacity(DATA_POINTS);
    for i in 0..DATA_POINTS {
        let point = lower + (upper - lower) * (i as f64) / ((DATA_POINTS - 1) as f64);
        inputs.push(point);
        outputs.push(output(point));
    }
    let data = GaussianProcessData::new(
        DMatrix::from_row_slice(1, DATA_POINTS, &inputs),
        DVector::from_vec(outputs),
        1e-10,
    )?;
    GaussianProcess::new(
        data,
        SquaredExponentialKernel::new(1.0, DVector::from_vec(vec![1.0]))?,
        state_dependency,
        control_dependency,
    )
}

fn known_dynamics_lipschitz() -> f64 {
    let trace = 2.25_f64;
    let determinant = 1.0_f64;
    let largest_eigenvalue = 0.5 * (trace + (trace.powi(2) - 4.0 * determinant).sqrt());
    largest_eigenvalue.sqrt()
}

fn rkhs_norm_bound() -> f64 {
    let c = 1.0;
    let beta = 0.5;
    let g1 = |t: f64| 0.1 * t.tanh();
    let dg1 = |t: f64| 0.1 * (1.0 - t.tanh().powi(2));
    let g2 = |t: f64| 0.1 * t.sin();
    let dg2 = |t: f64| 0.1 * t.cos();

    let n1 = rkhs_norm_squared(-3.0, 3.0, c, beta, g1, dg1).sqrt();
    let n2 = rkhs_norm_squared(-2.0, 2.0, c, beta, g2, dg2).sqrt();
    n1.max(n2)
}

fn rkhs_norm_squared(
    lower: f64,
    upper: f64,
    c: f64,
    beta: f64,
    g: impl Fn(f64) -> f64,
    dg: impl Fn(f64) -> f64,
) -> f64 {
    let boundary = (g(lower).powi(2) + g(upper).powi(2)) / (2.0 * c);
    let integral = simpson(lower, upper, 10_000, |t| {
        dg(t).powi(2) + beta.powi(2) * g(t).powi(2)
    });
    boundary + integral / (2.0 * beta * c)
}

fn simpson(lower: f64, upper: f64, intervals: usize, f: impl Fn(f64) -> f64) -> f64 {
    let n = if intervals % 2 == 0 {
        intervals
    } else {
        intervals + 1
    };
    let h = (upper - lower) / (n as f64);
    let mut sum = f(lower) + f(upper);
    for i in 1..n {
        let weight = if i % 2 == 0 { 2.0 } else { 4.0 };
        sum += weight * f(lower + h * (i as f64));
    }
    sum * h / 3.0
}

fn rounded(values: &[f64]) -> Vec<f64> {
    values
        .iter()
        .map(|value| (value * 1_000_000.0).round() / 1_000_000.0)
        .collect()
}

fn rounded_array(values: &[f64; HORIZON]) -> [f64; HORIZON] {
    values.map(|value| (value * 1_000_000.0).round() / 1_000_000.0)
}
