import rps.robotarium as robotarium
from rps.utilities.misc import *
from matplotlib.patches import Circle

import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp


# ============================================================
# Constant-forward-speed unicycle primitives:
#   1. keep heading
#   2. turn counterclockwise while moving
#   3. turn clockwise while moving
#
#   - Key/door identities are categorical hidden variables.
#   - Observations are probabilistic detections, not hard reveal/eliminate.
#   - Beliefs are updated by Bayes rule using q_y(y | z, position).
#   - Epistemic value uses expected entropy reduction under the same q_y.
# ============================================================

FS = 24

plt.rcParams.update({
    "font.family": "serif",
    "font.size": FS,
    "mathtext.fontset": "cm",        # Computer Modern (LaTeX-like)
    "axes.labelsize": FS,
    "axes.titlesize": FS,
    "xtick.labelsize": FS,
    "ytick.labelsize": FS,
    "legend.fontsize": FS,
    "figure.titlesize": FS,
})

# ============================================================
# Robotarium / workspace parameters
# ============================================================

N_ROBOTS = 1
SHOW_FIGURE = True
SIM_IN_REAL_TIME = True

RUN_EPISTEMIC = False

N_EPISODES = 1

BOUNDARY = np.array([-1.6, 1.6, -1.0, 1.0])
X_MIN, X_MAX, Y_MIN, Y_MAX = BOUNDARY

ROBOTARIUM_DT = 0.033

NUM_CANDIDATES = 5

# Planning lookahead time used only in the action scoring model.
# This can be larger than the physical step if you want the one-step score
# to see enough spatial displacement to change grid cells.
PLANNING_DT_MULTIPLIER = 20
DECISION_DT = ROBOTARIUM_DT * PLANNING_DT_MULTIPLIER

MAX_GATE_STEPS = 2000

# Grid used only for approximate finite-horizon planning.
N_X_BINS = 16
N_Y_BINS = 10
X_AXIS = np.linspace(X_MIN, X_MAX, N_X_BINS)
Y_AXIS = np.linspace(Y_MIN, Y_MAX, N_Y_BINS)

PLANNING_HORIZON = 8


# ============================================================
# Hidden key-door task defaults
# ============================================================

KEY_CANDIDATES = np.array([
    [-1.15, -0.70],
    [-1.05,  0.65],
    [-0.45,  0.00],
    [-0.45,  0.65],
    [ 0.10, -0.45],
    [ 0.0, -0.3],
])

DOOR_CANDIDATES = np.array([
    [ 0.70, -0.75],
    [ 0.80,  0.65],
    [ 1.25, -0.35],
    [ 1.25,  0.45],
    [ 0.45, -0.35],
    [ 0.6, 0.0],
])

TRUE_KEY_INDEX = 3
TRUE_DOOR_INDEX = 2

SENSOR_RADIUS = 0.3
PICKUP_RADIUS = 0.12
DOOR_RADIUS = PICKUP_RADIUS

INITIAL_CONDITION = np.array([
    [-1.35],
    [ 0.00],
    [ 0.00],
])


# ============================================================
# Observation model parameters
# ============================================================

# This is q_y. Detection is probabilistic and distance-dependent.
# If the true object is close, detection is likely but not certain.
# If no detection occurs, nearby candidates become less likely but are not
# eliminated exactly.
DETECTION_P_MAX = 1
DETECTION_SIGMA = SENSOR_RADIUS
DETECTION_EPS = 1e-8

# If True, picking up the key still collapses key belief exactly, because
# physical pickup confirms the true key.
COLLAPSE_ON_PICKUP = True

# ============================================================
# Cost parameters
# ============================================================
WALL_COST = 8.0
NON_TARGET_COST = 4
TARGET_SIGMA = SENSOR_RADIUS/2

SUCCESS_HOLD_STEPS = 1


# ============================================================
# Constant-forward-speed angular action grid
# ============================================================

FORWARD_SPEED = 0.2

OMEGA_MAX = 3.6 # max Robotarium rotation
N_OMEGA_ACTIONS = 37
OMEGA_GRID = np.linspace(-OMEGA_MAX, OMEGA_MAX, N_OMEGA_ACTIONS)
N_ACTIONS = len(OMEGA_GRID)

ACTION_PRIOR = np.ones(N_ACTIONS) / N_ACTIONS

OMEGA_TURN = 3.3
OMEGA_COV = 0.1 ** 2

PRIMITIVE_NAMES = [
    "go straight",
    "turn counterclockwise",
    "turn clockwise",
]
N_PRIMITIVES = len(PRIMITIVE_NAMES)


# ============================================================
# Basic utilities
# ============================================================

def wrap_angle(theta):
    return (theta + np.pi) % (2.0 * np.pi) - np.pi


def clip_position(x):
    return np.array([
        np.clip(x[0], X_MIN, X_MAX),
        np.clip(x[1], Y_MIN, Y_MAX),
    ])


def normalize(p):
    p = np.asarray(p, dtype=float)
    p = np.maximum(p, 1e-12)
    return p / np.sum(p)


def entropy(p):
    p = np.asarray(p, dtype=float)
    p = p[p > 1e-12]
    return -np.sum(p * np.log(p))


def gaussian_pdf_1d_grid(points, mean, var):
    diff = points - mean
    vals = np.exp(-0.5 * diff ** 2 / var)
    vals = np.maximum(vals, 1e-8)
    return vals / np.sum(vals)


# ============================================================
# Primitive library over angular velocity
# ============================================================

def make_primitives_on_omega_grid(omega_grid):
    """
    Primitive distributions over angular velocity.
    The forward speed is fixed, so the action is only omega.
        go straight:
            omega_mean = 0
        turn counterclockwise:
            omega_mean = +OMEGA_TURN
        turn clockwise:
            omega_mean = -OMEGA_TURN
    """
    primitive_means = [
        0.0,
        +OMEGA_TURN,
        -OMEGA_TURN,
    ]

    pi = np.zeros((N_PRIMITIVES, len(omega_grid)))

    for i, mean in enumerate(primitive_means):
        pi[i, :] = gaussian_pdf_1d_grid(
            omega_grid,
            mean,
            OMEGA_COV,
        )

    return pi


PI_OMEGA = make_primitives_on_omega_grid(OMEGA_GRID)


# ============================================================
# Belief and probabilistic observation model
# ============================================================

def initial_beliefs(key_candidates, door_candidates):
    b_key = np.ones(len(key_candidates)) / len(key_candidates)
    b_door = np.ones(len(door_candidates)) / len(door_candidates)
    return b_key, b_door


def detection_probability(position, candidate):
    """
    Distance-dependent probability of detecting the object if this candidate
    is the true object.

    This defines the soft range-limited observation model q_y.
    """
    d = np.linalg.norm(position - candidate)
    return DETECTION_P_MAX * np.exp(-0.5 * d ** 2 / DETECTION_SIGMA ** 2)


def sample_detection_observation(position, candidates, true_index, rng):
    """
        Sample the actual observation y from q_y(y | z=true_index, position).
        Observation space:
            y = None       means no object detected
            y = integer i  means candidate i detected
        If detection happens, the detected identity is the true candidate.
        """
    p_det = detection_probability(position, candidates[true_index])
    if rng.random() < p_det:
        return true_index
    return None


def observation_likelihood(observation, position, candidates):
    n = len(candidates)
    likelihood = np.zeros(n)

    for z in range(n):
        p_det = detection_probability(position, candidates[z])
        if observation is None:
            likelihood[z] = 1.0 - p_det
        elif z == observation:
            likelihood[z] = p_det
        else:
            likelihood[z] = 0

    return np.maximum(likelihood, DETECTION_EPS)


def predict_object_belief(belief, transition_matrix=None):
    """
    Prediction step for the categorical hidden variable.
    In this task, the true key and door identities are static, so the default
    transition is identity and the prediction is just a copy of the belief.
    """
    if transition_matrix is None:
        return belief.copy()
    return normalize(transition_matrix.T @ belief)


def correct_object_belief(belief_pred, candidates, true_index, position, rng):
    """
    Bayesian correction step for one object class.
        b_k(z) proportional to q_y(y_k | z, position) * b_pred(z)
    """
    observation = sample_detection_observation(position, candidates, true_index, rng)
    likelihood = observation_likelihood(observation, position, candidates)
    new_belief = normalize(belief_pred * likelihood)
    return new_belief, observation


def update_beliefs(
    b_key,
    b_door,
    carrying_key,
    position,
    key_candidates,
    door_candidates,
    true_key_index,
    true_door_index,
    rng
):
    """
    Full perception update.
    This implements the predict-correct structure:
        b_pred = q_x prediction
        b_new  = Bayesian correction with q_y
    For key/door identities, q_x is identity because the true key/door do not move between time steps.
    """
    key_observation = None
    door_observation = None

    if not carrying_key:
        b_key_pred = predict_object_belief(b_key)
        b_key, key_observation = correct_object_belief(
            b_key_pred,
            key_candidates,
            true_key_index,
            position,
            rng
        )

    b_door_pred = predict_object_belief(b_door)
    b_door, door_observation = correct_object_belief(
        b_door_pred,
        door_candidates,
        true_door_index,
        position,
        rng
    )

    return b_key, b_door, key_observation, door_observation


# ============================================================
# Expected information gain under the same soft observation model
# ============================================================

def object_information_gain(position, belief, candidates):
    """
    Expected entropy reduction:
        IG = H[b(z)] - E_{y ~ q_y(y | b, position)} H[b(z | y)]
    This uses the same q_y as the online belief update.
    """
    prior_H = entropy(belief)
    n = len(candidates)

    # All possible observations: None (∅) plus each candidate index.
    observations = [None] + list(range(n))

    expected_H_post = 0.0   # initialize

    for obs in observations:
        likelihood = observation_likelihood(obs, position, candidates)
        p_obs = np.sum(belief * likelihood)

        if p_obs > 1e-12:
            posterior = normalize(belief * likelihood)
            expected_H_post += p_obs * entropy(posterior)

    ig = prior_H - expected_H_post
    return max(0.0, ig)


def epistemic_value(
    position,
    b_key,
    b_door,
    carrying_key,
    key_candidates,
    door_candidates,
):
    ig = 0.0

    if not carrying_key:
        ig += object_information_gain(position, b_key, key_candidates)
    else:
        ig += object_information_gain(position, b_door, door_candidates)

    return ig


# ============================================================
# Instrumental cost
# ============================================================
def compute_boundary_cost(x, y, boundary_points):
    cost_b_term = 0.0
    var_b = 0.15 ** 2

    for b_ind in range(2):
        dist_vec_x = np.abs(x - boundary_points[b_ind])
        dist_vec_y = np.abs(y - boundary_points[b_ind + 2])
        cost_b_term += np.exp(-0.5 / var_b * dist_vec_x ** 2)
        cost_b_term += np.exp(-0.5 / var_b * dist_vec_y ** 2)

    return cost_b_term


def target_score(position, b_key, b_door, carrying_key, key_candidates, door_candidates):
    if not carrying_key:
        candidates = key_candidates
        belief = b_key
    else:
        candidates = door_candidates
        belief = b_door

    d2 = np.sum((candidates - position.reshape(1, 2)) ** 2, axis=1)
    likelihood = np.exp(-0.5 * d2 / (TARGET_SIGMA ** 2))

    score = np.sum(belief * likelihood)

    return score



def instrumental_cost(
    position,
    omega,
    b_key,
    b_door,
    carrying_key,
    key_candidates,
    door_candidates,
):

    score = target_score(
        position,
        b_key,
        b_door,
        carrying_key,
        key_candidates,
        door_candidates,
    )

    state_cost = NON_TARGET_COST * (- score)

    wall_cost = WALL_COST * compute_boundary_cost(
        position[0],
        position[1],
        BOUNDARY,
    )

    return state_cost + wall_cost


# ============================================================
# Unicycle prediction
# ============================================================

def unicycle_position_step(position, theta, omega, dt):
    """
    Predict position after dt with constant forward speed and angular velocity.
    """
    x, y = position

    if abs(omega) < 1e-8:
        x_next = x + FORWARD_SPEED * dt * np.cos(theta)
        y_next = y + FORWARD_SPEED * dt * np.sin(theta)
    else:
        theta_next = theta + omega * dt
        x_next = x + (FORWARD_SPEED / omega) * (
            np.sin(theta_next) - np.sin(theta)
        )
        y_next = y - (FORWARD_SPEED / omega) * (
            np.cos(theta_next) - np.cos(theta)
        )

    return clip_position(np.array([x_next, y_next]))


# ============================================================
# Finite-horizon value approximation over position grid
# ============================================================

def nearest_grid_index(position):
    ix = int(np.argmin(np.abs(X_AXIS - position[0])))
    iy = int(np.argmin(np.abs(Y_AXIS - position[1])))
    return ix, iy


def grid_position(ix, iy):
    return np.array([X_AXIS[ix], Y_AXIS[iy]])


def grid_neighbors(ix, iy):
    neighbors = []

    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue

            jx = np.clip(ix + dx, 0, N_X_BINS - 1)
            jy = np.clip(iy + dy, 0, N_Y_BINS - 1)

            neighbors.append((int(jx), int(jy)))

    return list(set(neighbors))


def compute_epistemic_grid(
    b_key,
    b_door,
    carrying_key,
    key_candidates,
    door_candidates,
):
    E = np.zeros((N_X_BINS, N_Y_BINS))

    for ix in range(N_X_BINS):
        for iy in range(N_Y_BINS):
            pos = grid_position(ix, iy)
            E[ix, iy] = epistemic_value(
                pos,
                b_key,
                b_door,
                carrying_key,
                key_candidates,
                door_candidates,
            )

    return E


def compute_value_grid(
    b_key,
    b_door,
    carrying_key,
    key_candidates,
    door_candidates,
    use_epistemic=True,
    epistemic_grid=None,
    horizon=PLANNING_HORIZON,
):
    V = np.zeros((N_X_BINS, N_Y_BINS))

    for _ in range(horizon):
        V_new = np.zeros_like(V)

        for ix in range(N_X_BINS):
            for iy in range(N_Y_BINS):
                values = []

                for jx, jy in grid_neighbors(ix, iy):
                    pos_next = grid_position(jx, jy)

                    inst = instrumental_cost(
                        pos_next,
                        0.0,
                        b_key,
                        b_door,
                        carrying_key,
                        key_candidates,
                        door_candidates,
                    )

                    if use_epistemic:
                        epi = epistemic_grid[jx, jy]
                    else:
                        epi = 0.0

                    values.append(inst - epi + V[jx, jy])

                V_new[ix, iy] = np.min(values)

        V = V_new

    return V


def value_at_position(V, position):
    ix, iy = nearest_grid_index(position)
    return V[ix, iy]


def action_scores_from_value(
    position,
    theta,
    b_key,
    b_door,
    carrying_key,
    key_candidates,
    door_candidates,
    use_epistemic=True,
):
    if use_epistemic:
        epistemic_grid = compute_epistemic_grid(
            b_key,
            b_door,
            carrying_key,
            key_candidates,
            door_candidates,
        )
    else:
        epistemic_grid = None

    V = compute_value_grid(
        b_key,
        b_door,
        carrying_key,
        key_candidates,
        door_candidates,
        use_epistemic=use_epistemic,
        epistemic_grid=epistemic_grid,
        horizon=PLANNING_HORIZON,
    )

    scores = np.zeros(N_ACTIONS)

    for j, omega in enumerate(OMEGA_GRID):
        pos_next = unicycle_position_step(
            position,
            theta,
            omega,
            DECISION_DT,
        )

        inst = instrumental_cost(
            pos_next,
            omega,
            b_key,
            b_door,
            carrying_key,
            key_candidates,
            door_candidates,
        )

        if use_epistemic:
            epi = epistemic_value(
                pos_next,
                b_key,
                b_door,
                carrying_key,
                key_candidates,
                door_candidates,
            )
        else:
            epi = 0.0

        future = value_at_position(V, pos_next)
        scores[j] = inst - epi + future

    return scores, V


# ============================================================
# BeFree-Gate optimizer
# ============================================================
def solve_befree_gate(action_scores, pi_omega):
    # Use raw scores if you want absolute cost scale to matter.
    # Use normalized scores if the optimization becomes numerically too sharp.
    scores = action_scores
    # scores = normalize_scores(action_scores)

    w = cp.Variable(N_PRIMITIVES)
    p_comb = pi_omega.T @ w

    constraints = [
        w >= 0,
        cp.sum(w) == 1
    ]

    objective = (
        scores @ p_comb
        + cp.sum(cp.rel_entr(p_comb, ACTION_PRIOR))
    )

    problem = cp.Problem(cp.Minimize(objective), constraints)

    problem.solve(
        solver=cp.SCS,
        verbose=False,
        max_iters=1000,
        eps=1e-3,
        warm_start=True,
    )

    if w.value is None:
        raise RuntimeError("CVX optimization failed.")

    w_opt = np.maximum(w.value, 0.0)
    w_opt = w_opt / np.sum(w_opt)

    p_action = pi_omega.T @ w_opt
    p_action = np.maximum(p_action, 0.0)
    p_action = p_action / np.sum(p_action)

    return w_opt, p_action


def sample_omega(p_action, rng, deterministic=False):
    if deterministic:
        idx = int(np.argmax(p_action))
    else:
        idx = rng.choice(N_ACTIONS, p=p_action)

    return OMEGA_GRID[idx], idx


# ============================================================
# Robotarium unicycle control
# ============================================================

def apply_unicycle_velocity(r, v, omega):
    dxu = np.array([
        [v],
        [omega],
    ])

    dxu_limited = r._threshold(dxu)

    r.set_velocities(np.arange(N_ROBOTS), dxu_limited)
    r.step()


# ============================================================
# Plotting in Robotarium
# ============================================================

def add_landmark_markers(r, key_candidates, door_candidates, true_key, true_door):
    ax = r._axes_handle

    for p in key_candidates:
        ax.add_patch(
            Circle(
                (p[0], p[1]),
                radius=PICKUP_RADIUS,
                fill=False,
                edgecolor="orange",
                linewidth=2,
                zorder=-2,
            )
        )

    for p in door_candidates:
        ax.add_patch(
            Circle(
                (p[0], p[1]),
                radius=DOOR_RADIUS,
                fill=False,
                edgecolor="purple",
                linewidth=2,
                zorder=-2,
            )
        )

    ax.scatter(
        true_key[0],
        true_key[1],
        marker="*",
        color="orange",
        edgecolor="black",
        s=400,
        label="true key",
    )

    ax.scatter(
        true_door[0],
        true_door[1],
        marker="D",
        color="purple",
        edgecolor="black",
        s=200,
        label="true door",
    )


def add_sensor_marker(r, position):
    ax = r._axes_handle

    sensor_marker = Circle(
        (position[0], position[1]),
        radius=SENSOR_RADIUS,
        fill=False,
        edgecolor="green",
        linewidth=1.5,
        linestyle="--",
        zorder=1,
    )

    ax.add_patch(sensor_marker)
    return sensor_marker


def update_sensor_marker(r, sensor_marker, position):
    sensor_marker.center = (position[0], position[1])


# ============================================================
# Randomized episode configuration
# ============================================================

def sample_robotarium_start(
    rng,
    true_key,
    true_door,
    margin=SENSOR_RADIUS + 0.05,
    avoid_visible=True,
):
    for _ in range(10_000):
        x = rng.uniform(X_MIN + margin, X_MAX - margin)
        y = rng.uniform(Y_MIN + margin, Y_MAX - margin)
        pos = np.array([x, y])

        if np.linalg.norm(pos - true_key) <= PICKUP_RADIUS + margin:
            continue

        if np.linalg.norm(pos - true_door) <= DOOR_RADIUS + margin:
            continue

        if avoid_visible:
            if np.linalg.norm(pos - true_key) <= margin:
                continue
            if np.linalg.norm(pos - true_door) <= margin:
                continue

        theta = rng.uniform(-np.pi, np.pi)

        return np.array([
            [x],
            [y],
            [theta],
        ])

    raise RuntimeError("Could not sample a valid initial condition.")


def sample_candidate_positions(
    rng,
    n_key_candidates=NUM_CANDIDATES,
    n_door_candidates=NUM_CANDIDATES,
    margin=0.25,
    min_candidate_separation=0.35,
):
    """
    Sample key and door candidate locations for one episode.

    Keys are sampled mostly on the left side.
    Doors are sampled mostly on the right side.
    This preserves the key-then-door task structure while changing locations.
    """
    points = []

    def far_enough(p, existing):
        for q in existing:
            if np.linalg.norm(p - q) < min_candidate_separation:
                return False
        return True

    key_candidates = []

    for _ in range(10_000):
        if len(key_candidates) >= n_key_candidates:
            break

        x = rng.uniform(X_MIN + margin, 0.20)
        y = rng.uniform(Y_MIN + margin, Y_MAX - margin)
        p = np.array([x, y])

        if far_enough(p, points):
            key_candidates.append(p)
            points.append(p)

    door_candidates = []

    for _ in range(10_000):
        if len(door_candidates) >= n_door_candidates:
            break

        x = rng.uniform(-0.20, X_MAX - margin)
        y = rng.uniform(Y_MIN + margin, Y_MAX - margin)
        p = np.array([x, y])

        if far_enough(p, points):
            door_candidates.append(p)
            points.append(p)

    if len(key_candidates) < n_key_candidates:
        raise RuntimeError("Could not sample enough key candidates.")

    if len(door_candidates) < n_door_candidates:
        raise RuntimeError("Could not sample enough door candidates.")

    return np.array(key_candidates), np.array(door_candidates)


def sample_robotarium_episode_configs(n_episodes, seed):
    rng = np.random.default_rng(seed)
    configs = []

    for ep in range(n_episodes):
        key_candidates, door_candidates = sample_candidate_positions(
            rng,
            n_key_candidates=NUM_CANDIDATES,
            n_door_candidates=NUM_CANDIDATES,
            margin=0.25,
            min_candidate_separation=0.35,
        )

        true_key_index = int(rng.integers(len(key_candidates)))
        true_door_index = int(rng.integers(len(door_candidates)))

        true_key = key_candidates[true_key_index]
        true_door = door_candidates[true_door_index]

        initial_condition = sample_robotarium_start(
            rng,
            true_key=true_key,
            true_door=true_door,
            margin=SENSOR_RADIUS + 0.05,
            avoid_visible=True,
        )

        configs.append({
            "key_candidates": key_candidates,
            "door_candidates": door_candidates,
            "true_key_index": true_key_index,
            "true_door_index": true_door_index,
            "initial_condition": initial_condition,
            "seed": seed + ep,
        })

    return configs


# ============================================================
# One Robotarium episode
# ============================================================

def run_robotarium_episode(
    ep,
    use_epistemic=True,
    deterministic=False,
    initial_condition=None,
    key_candidates=None,
    door_candidates=None,
    true_key_index=TRUE_KEY_INDEX,
    true_door_index=TRUE_DOOR_INDEX,
    seed=None,
):
    if key_candidates is None:
        key_candidates = KEY_CANDIDATES.copy()
    else:
        key_candidates = np.array(key_candidates, dtype=float, copy=True)

    if door_candidates is None:
        door_candidates = DOOR_CANDIDATES.copy()
    else:
        door_candidates = np.array(door_candidates, dtype=float, copy=True)

    if initial_condition is None:
        initial_condition = INITIAL_CONDITION.copy()
    else:
        initial_condition = np.array(initial_condition, dtype=float, copy=True)

    true_key = key_candidates[true_key_index]
    true_door = door_candidates[true_door_index]

    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    r = robotarium.Robotarium(
        number_of_robots=N_ROBOTS,
        show_figure=SHOW_FIGURE,
        initial_conditions=initial_condition.copy(),
        sim_in_real_time=SIM_IN_REAL_TIME,
        skip_initialization=True
    )

    add_landmark_markers(r, key_candidates, door_candidates, true_key, true_door)

    b_key, b_door = initial_beliefs(key_candidates, door_candidates)

    carrying_key = False
    success = False

    trajectory = []
    weights_history = []
    omega_history = []
    key_entropy_history = []
    door_entropy_history = []
    theta_history = []
    key_observation_history = []
    door_observation_history = []
    success_hold_counter = 0
    key_visible_time = None
    pickup_time = None
    door_visible_time = None
    success_time = None

    x_uni = r.get_poses()
    position = x_uni[:2, 0]
    theta = wrap_angle(x_uni[2, 0])

    sensor_marker = add_sensor_marker(r, position)

    b_key, b_door, key_obs, door_obs = update_beliefs(
        b_key,
        b_door,
        carrying_key,
        position,
        key_candidates,
        door_candidates,
        true_key_index,
        true_door_index,
        rng
    )

    key_observation_history.append(key_obs)
    door_observation_history.append(door_obs)

    for gate_step in range(MAX_GATE_STEPS):
        trajectory.append(position.copy())
        key_entropy_history.append(entropy(b_key))
        door_entropy_history.append(entropy(b_door))
        theta_history.append(theta)

        # Door success condition.
        if carrying_key and np.linalg.norm(position - true_door) <= DOOR_RADIUS:
            success_hold_counter += 1
            success_time = gate_step
            print("Door accessed at step", success_time)
            if success_hold_counter >= SUCCESS_HOLD_STEPS:
                success = True
                break
        else:
            success_hold_counter = 0

        action_scores, V = action_scores_from_value(
            position,
            theta,
            b_key,
            b_door,
            carrying_key,
            key_candidates,
            door_candidates,
            use_epistemic=use_epistemic,
        )

        w_opt, p_action = solve_befree_gate(
            action_scores,
            PI_OMEGA,
        )

        omega_cmd, action_index = sample_omega(
            p_action,
            rng,
            deterministic=deterministic,
        )

        weights_history.append(w_opt.copy())
        omega_history.append(omega_cmd)

        apply_unicycle_velocity(
            r,
            FORWARD_SPEED,
            omega_cmd,
        )
        x_uni = r.get_poses()

        position = x_uni[:2, 0]
        theta = wrap_angle(x_uni[2, 0])

        update_sensor_marker(r, sensor_marker, position)

        if key_visible_time is None and key_obs == true_key_index:
            key_visible_time = gate_step + 1

        if door_visible_time is None and door_obs == true_door_index:
            door_visible_time = gate_step + 1

        # Physical pickup after movement confirms the key.
        if (not carrying_key) and np.linalg.norm(position - true_key) <= PICKUP_RADIUS:
            carrying_key = True
            pickup_time = gate_step + 1
            if COLLAPSE_ON_PICKUP:
                b_key = np.zeros_like(b_key)
                b_key[true_key_index] = 1.0
            print("Key picked up at step", pickup_time)

        # Soft Bayesian observation update after movement.
        b_key, b_door, key_obs, door_obs = update_beliefs(
            b_key,
            b_door,
            carrying_key,
            position,
            key_candidates,
            door_candidates,
            true_key_index,
            true_door_index,
            rng
        )

        key_observation_history.append(key_obs)
        door_observation_history.append(door_obs)

    apply_unicycle_velocity(
        r,
        0.0,
        0.0,
    )

    r.debug()

    trajectory = np.array(trajectory)
    weights_history = np.array(weights_history)
    omega_history = np.array(omega_history)
    key_entropy_history = np.array(key_entropy_history)
    door_entropy_history = np.array(door_entropy_history)
    theta_history = np.array(theta_history)

    return {
        "success": success,
        "trajectory": trajectory,
        "weights": weights_history,
        "omega": omega_history,
        "theta": theta_history,
        "key_entropy": key_entropy_history,
        "door_entropy": door_entropy_history,
        "key_observations": key_observation_history,
        "door_observations": door_observation_history,
        "final_key_belief": b_key,
        "final_door_belief": b_door,
        "key_visible_time": key_visible_time,
        "pickup_time": pickup_time,
        "door_visible_time": door_visible_time,
        "door_access_time": success_time,
        "key_candidates": key_candidates,
        "door_candidates": door_candidates,
        "true_key_index": true_key_index,
        "true_door_index": true_door_index,
        "true_key": true_key,
        "true_door": true_door,
        "initial_condition": initial_condition.copy(),
        "seed": seed,
    }


# ============================================================
# Multi-episode evaluation
# ============================================================

def run_robotarium_configs(
    configs,
    use_epistemic=True,
    deterministic=False,
    prefix="robotarium",
):
    results = []

    for ep, cfg in enumerate(configs):
        result = run_robotarium_episode(
            ep,
            use_epistemic=use_epistemic,
            deterministic=deterministic,
            initial_condition=cfg["initial_condition"].copy(),
            key_candidates=cfg["key_candidates"].copy(),
            door_candidates=cfg["door_candidates"].copy(),
            true_key_index=cfg["true_key_index"],
            true_door_index=cfg["true_door_index"],
            seed=cfg["seed"],
        )

        result.update(cfg)
        results.append(result)

    success_rate = np.mean([r["success"] for r in results])

    key_times = [
        r["key_visible_time"] if r["key_visible_time"] is not None else MAX_GATE_STEPS
        for r in results
    ]

    pickup_times = [
        r["pickup_time"] if r["pickup_time"] is not None else MAX_GATE_STEPS
        for r in results
    ]

    door_times = [
        r["door_visible_time"] if r["door_visible_time"] is not None else MAX_GATE_STEPS
        for r in results
    ]

    gate_steps = [
        len(r["trajectory"])
        for r in results
    ]

    stats = {
        "results": results,
        "success_rate": success_rate,
        "mean_gate_steps": np.mean(gate_steps),
        "std_gate_steps": np.std(gate_steps),
        "mean_key_visible": np.mean(key_times),
        "std_key_visible": np.std(key_times),
        "mean_pickup": np.mean(pickup_times),
        "std_pickup": np.std(pickup_times),
        "mean_door_visible": np.mean(door_times),
        "std_door_visible": np.std(door_times),
        "use_epistemic": use_epistemic,
        "prefix": prefix,
    }

    return stats


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    configs = sample_robotarium_episode_configs(
        n_episodes=N_EPISODES,
        seed=1000,
    )

    stats_epi = run_robotarium_configs(
        configs,
        use_epistemic=RUN_EPISTEMIC,
        deterministic=False,
        prefix="robotarium_epi",
    )
