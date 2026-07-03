//! Fixed-interval RTS smoother for the 2-state wheel chain `{w, ẇ}`.
//!
//! **Why this exists.** The suspension estimator's forward IEKF (`run()` in
//! `run.rs`) propagates each wheel's travel/velocity as a double integrator driven
//! by a precomputed differential-acceleration control and anchored by occasional
//! event factors (airborne topout `w=0`, stationary zero-wheel-velocity, soft
//! `[0, max]` barrier, optional sag prior). In the 24-DOF joint filter each wheel
//! block is **exactly decoupled** — the transition Jacobian, process noise, and
//! initial covariance are block-diagonal, and every non-wheel factor has zero
//! Jacobian on the wheel columns — so each wheel's marginal can be reproduced by
//! the standalone 2-state forward filter in [`forward_filter`].
//!
//! A forward filter alone corrects travel *after* an anchor arrives (the estimate
//! drifts during unconstrained riding, then snaps at the topout). The
//! **fixed-interval RTS backward pass** (Bell 1994: forward KF + RTS = the batch
//! MAP solve) distributes each anchor's information backward over the interval
//! *before* it, turning consecutive topouts into two-sided boundary conditions on
//! the travel integral.
//!
//! State: `[w, ẇ]` (travel m, velocity m/s). Covariance stored upper-triangular as
//! `(p00, p01, p11)`. Memory: O(n) at 5 f64 per sample per wheel.
//!
//! Reference: Bell, B. M. & Cathey, F. W. (1994). "The iterated Kalman smoother as
//! a Gauss-Newton method." *IEEE Trans. Autom. Control*, 39(10), 2153–2156.

/// Tuning for one wheel's standalone 2-state pass.
///
/// Values mirror those the 24-DOF forward filter uses for the same block (see
/// `run.rs`/`process.rs`), so [`forward_filter`] reproduces `run()`'s wheel
/// marginal exactly at `iekf_iters = 1` (the default operating point).
#[derive(Debug, Clone, Copy)]
pub struct WheelParams {
    /// Sample period, s.
    pub dt: f64,
    /// Travel process noise variance per step, m²  (= `wheel_pos_rw² · dt`).
    pub q_pos: f64,
    /// Velocity process noise variance per step, (m/s)²  (= `wheel_vel_rw² · dt`).
    pub q_vel: f64,
    /// Initial travel variance, m².
    pub init_travel_var: f64,
    /// Initial velocity variance, (m/s)².
    pub init_vel_var: f64,
    /// Zero-wheel-velocity pseudo-measurement std, m/s.
    pub zupt_sigma: f64,
    /// Airborne topout reference std, m.
    pub topout_sigma: f64,
    /// Soft `[0, travel_max]` barrier std, m.
    pub barrier_sigma: f64,
    /// Sag prior std, m (used only when `use_sag_prior` is `true`).
    pub sag_sigma: f64,
    /// Sag target travel, m.
    pub sag: f64,
    /// Maximum travel, m.
    pub travel_max: f64,
    /// Fire the sag prior on non-stationary, non-airborne samples.
    pub use_sag_prior: bool,
}

/// One filtered sample: Gaussian belief `[w, ẇ]` after one sample's full factor
/// sequence. Travel `w` in m; velocity `wv` in m/s; covariance upper triangle in
/// m² / (m·m/s) / (m/s)².
#[derive(Debug, Clone, Copy)]
pub struct FilteredSample {
    /// Travel, m.
    pub w: f64,
    /// Wheel velocity, m/s.
    pub wv: f64,
    /// Travel variance, m².
    pub p00: f64,
    /// Travel–velocity cross-covariance, m·(m/s).
    pub p01: f64,
    /// Velocity variance, (m/s)².
    pub p11: f64,
}

/// One smoothed sample (backward pass output). Travel variance `p00` retained for
/// diagnostics and the test in §4 (smoothed ≤ filtered).
#[derive(Debug, Clone, Copy)]
pub struct SmoothedSample {
    /// Smoothed travel, m.
    pub w: f64,
    /// Smoothed wheel velocity, m/s.
    pub wv: f64,
    /// Smoothed travel variance, m².
    pub p00: f64,
}

// ---------------------------------------------------------------------------
// Internal linear algebra helpers (plain f64, no nalgebra)
// ---------------------------------------------------------------------------

/// Multiplies 2×2 symmetric matrix `P = [[p00, p01], [p01, p11]]` by 2×2 `F`
/// from the left: returns `F · P`.
#[inline]
fn mat2_mul_fp(f00: f64, f01: f64, f10: f64, f11: f64, p00: f64, p01: f64, p11: f64)
    -> (f64, f64, f64, f64)
{
    // FP row 0: [f00*p00+f01*p10,  f00*p01+f01*p11]  (p10 = p01 by symmetry)
    // FP row 1: [f10*p00+f11*p10,  f10*p01+f11*p11]
    let fp00 = f00 * p00 + f01 * p01;
    let fp01 = f00 * p01 + f01 * p11;
    let fp10 = f10 * p00 + f11 * p01;
    let fp11 = f10 * p01 + f11 * p11;
    (fp00, fp01, fp10, fp11)
}

/// `F P Fᵀ` for `F = [[1, dt], [0, 1]]` and symmetric `P`.
/// Returns the new symmetric covariance as `(p00, p01, p11)`.
#[inline]
fn predict_cov(p00: f64, p01: f64, p11: f64, dt: f64, q_pos: f64, q_vel: f64) -> (f64, f64, f64) {
    // FP (non-symmetric intermediate):
    let (fp00, fp01, _fp10, fp11) = mat2_mul_fp(1.0, dt, 0.0, 1.0, p00, p01, p11);
    // FP Fᵀ = FP · [[1,0],[dt,1]] row by row.
    // [0,0]: fp00 + fp01·dt   [0,1]: fp01   [1,1]: fp11
    // [1,0] = fp10 + fp11·dt  but equals [0,1] by symmetry (P symmetric → FPFᵀ symmetric).
    let new_p00 = fp00 + fp01 * dt;
    let new_p01 = fp01;
    let new_p11 = fp11;
    (new_p00 + q_pos, new_p01, new_p11 + q_vel)
}

/// Closed-form inverse of a 2×2 symmetric positive-definite matrix.
/// Returns `None` if `det ≤ threshold`.
#[inline]
fn inv2(a00: f64, a01: f64, a11: f64) -> Option<(f64, f64, f64)> {
    let det = a00 * a11 - a01 * a01;
    if det <= 1e-300 {
        return None;
    }
    let inv_det = 1.0 / det;
    Some((a11 * inv_det, -a01 * inv_det, a00 * inv_det))
}

// ---------------------------------------------------------------------------
// Scalar measurement update  (sign convention: r = z ⊟ h(x), H = ∂r/∂δ)
// ---------------------------------------------------------------------------

/// Applies a scalar Kalman update in-place. `h0`, `h1` are the two columns of the
/// 1×2 measurement Jacobian `H`. `r` is the residual `z − h(x)`. `r_var` is the
/// measurement noise variance R.
///
/// Joseph-form: `P ← (I − KH) P (I − KH)ᵀ + K R Kᵀ`.
///
/// Returns `false` and leaves the state unchanged if the innovation covariance
/// `S = H P Hᵀ + R` is non-positive or any output is non-finite — degenerate
/// factors are skipped, never poisoning the pass.
#[inline]
fn scalar_update(
    w: &mut f64, wv: &mut f64,
    p00: &mut f64, p01: &mut f64, p11: &mut f64,
    h0: f64, h1: f64, r: f64, r_var: f64,
) -> bool {
    // S = H P Hᵀ + R  (scalar)
    let ph0 = h0 * *p00 + h1 * *p01; // (PH^T)_0  — column 0 of PH^T
    let ph1 = h0 * *p01 + h1 * *p11; // (PH^T)_1  — column 1 of PH^T
    let s = h0 * ph0 + h1 * ph1 + r_var;
    if s <= 0.0 {
        return false;
    }
    // K = P Hᵀ S⁻¹   (2×1)
    let k0 = ph0 / s;
    let k1 = ph1 / s;
    // Mean update: δ = −K·r — the single-iteration collapse of the engine's
    // `δ = K(Hδᵢ − r)` at δᵢ = 0 (iekf.rs), under H = ∂r/∂δ. (With H carrying the
    // −1 entries, −K·r equals the textbook +K_std·(z − h(x)) correction.)
    let dw  = -(k0 * r);
    let dwv = -(k1 * r);
    let new_w  = *w  + dw;
    let new_wv = *wv + dwv;
    if !new_w.is_finite() || !new_wv.is_finite() {
        return false;
    }
    // Joseph form: (I − KH) P (I − KH)ᵀ + K R Kᵀ
    // (I − KH) = [[1−k0·h0, −k0·h1], [−k1·h0, 1−k1·h1]]
    let a00 = 1.0 - k0 * h0;
    let a01 =     - k0 * h1;
    let a10 =     - k1 * h0;
    let a11 = 1.0 - k1 * h1;
    // A P Aᵀ  (A = I − KH)
    let (ap00, ap01, ap10, ap11) = mat2_mul_fp(a00, a01, a10, a11, *p00, *p01, *p11);
    // (AP)·Aᵀ with Aᵀ[j][k] = A[k][j]:
    //   [0,0] = ap00·a00 + ap01·a01,  [0,1] = ap00·a10 + ap01·a11,
    //   [1,1] = ap10·a10 + ap11·a11.
    let new_p00 = ap00 * a00 + ap01 * a01;
    let new_p01 = ap00 * a10 + ap01 * a11;
    let new_p11 = ap10 * a10 + ap11 * a11;
    // + K R K^T
    let new_p00 = new_p00 + k0 * r_var * k0;
    let new_p01 = new_p01 + k0 * r_var * k1;
    let new_p11 = new_p11 + k1 * r_var * k1;
    if !new_p00.is_finite() || !new_p01.is_finite() || !new_p11.is_finite() {
        return false;
    }
    *w   = new_w;
    *wv  = new_wv;
    *p00 = new_p00;
    *p01 = new_p01;
    *p11 = new_p11;
    true
}

// ---------------------------------------------------------------------------
// 2×2 barrier update  (joint 2-row update — matches the engine's 2×2 solve)
// ---------------------------------------------------------------------------

/// Applies the joint 2×2 barrier update for the soft `[0, travel_max]` bounds.
///
/// `H` is a 2×2 matrix given row-by-row: `h_row0 = [h00, h01]`, `h_row1 = [h10, h11]`.
/// `r = [r0, r1]` is the 2-vector residual. `r_var` is the shared scalar barrier
/// variance (R = r_var · I₂).
///
/// Short-circuits (no-op) when both residual rows and both H rows are zero —
/// provably output-preserving, no wasted work on in-band samples.
///
/// Returns `false` and leaves state unchanged on numerical failure.
#[allow(clippy::too_many_arguments)]
#[inline]
fn barrier_update(
    w: &mut f64, wv: &mut f64,
    p00: &mut f64, p01: &mut f64, p11: &mut f64,
    h00: f64, h01: f64, h10: f64, h11: f64,
    r0: f64, r1: f64, r_var: f64,
) -> bool {
    // Short-circuit when all H rows and residuals are zero (in-band sample)
    if h00 == 0.0 && h01 == 0.0 && h10 == 0.0 && h11 == 0.0
        && r0 == 0.0 && r1 == 0.0
    {
        return true; // no-op, not a failure
    }

    // S = H P Hᵀ + R·I₂  (2×2 symmetric)
    // PHᵀ[state i][meas j] = P[i,:]·H[j,:]  (Hᵀ column j is H row j):
    let ph00 = *p00 * h00 + *p01 * h01; // state w,  meas 0
    let ph01 = *p00 * h10 + *p01 * h11; // state w,  meas 1
    let ph10 = *p01 * h00 + *p11 * h01; // state ẇ, meas 0
    let ph11 = *p01 * h10 + *p11 * h11; // state ẇ, meas 1
    let s00 = h00 * ph00 + h01 * ph10 + r_var;
    let s01 = h00 * ph01 + h01 * ph11;
    let s11 = h10 * ph01 + h11 * ph11 + r_var;

    // Invert S (2×2 symmetric positive definite)
    let (si00, si01, si11) = match inv2(s00, s01, s11) {
        Some(inv) => inv,
        None => return false,
    };

    // K = PH^T S^{-1}  (2×2: rows = state, cols = measurements)
    // k[state_row][meas_col]
    let k00 = ph00 * si00 + ph01 * si01;
    let k01 = ph00 * si01 + ph01 * si11;
    let k10 = ph10 * si00 + ph11 * si01;
    let k11 = ph10 * si01 + ph11 * si11;

    // Mean update: δ = −K·r (see `scalar_update` — the engine's GN step at δᵢ = 0).
    let dw  = -(k00 * r0 + k01 * r1);
    let dwv = -(k10 * r0 + k11 * r1);
    let new_w  = *w  + dw;
    let new_wv = *wv + dwv;
    if !new_w.is_finite() || !new_wv.is_finite() {
        return false;
    }

    // Joseph form: (I − KH) P (I − KH)ᵀ + K R Kᵀ
    // A = I − KH  (2×2)
    let a00 = 1.0 - k00 * h00 - k01 * h10;
    let a01 =     - k00 * h01 - k01 * h11;
    let a10 =     - k10 * h00 - k11 * h10;
    let a11 = 1.0 - k10 * h01 - k11 * h11;

    // A P Aᵀ — (AP)·Aᵀ with Aᵀ[j][k] = A[k][j] (A is NOT symmetric).
    let (ap00, ap01, ap10, ap11) = mat2_mul_fp(a00, a01, a10, a11, *p00, *p01, *p11);
    let new_p00 = ap00 * a00 + ap01 * a01;
    let new_p01 = ap00 * a10 + ap01 * a11;
    let new_p11 = ap10 * a10 + ap11 * a11;

    // + K R K^T  (R = r_var·I₂, so K R Kᵀ = r_var · K Kᵀ)
    let new_p00 = new_p00 + r_var * (k00 * k00 + k01 * k01);
    let new_p01 = new_p01 + r_var * (k00 * k10 + k01 * k11);
    let new_p11 = new_p11 + r_var * (k10 * k10 + k11 * k11);

    if !new_p00.is_finite() || !new_p01.is_finite() || !new_p11.is_finite() {
        return false;
    }

    *w   = new_w;
    *wv  = new_wv;
    *p00 = new_p00;
    *p01 = new_p01;
    *p11 = new_p11;
    true
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Forward Kalman filter for one wheel's 2-state `{w, ẇ}` double integrator.
///
/// Mirrors the per-sample wheel sequence of `run()`'s IEKF at `iekf_iters = 1`
/// exactly:
/// 1. **Predict** with `F = [[1, dt], [0, 1]]`, `Q = diag(q_pos, q_vel)`.
/// 2. ZUPT if `stationary[i]`.
/// 3. Airborne topout + ZUPT if `airborne[i]`; otherwise sag prior if
///    `use_sag_prior && !stationary[i]`.
/// 4. Joint 2×2 barrier update `[0, travel_max]` (no-op when in-band).
///
/// Update residuals follow the convention `r = z ⊟ h(x)`, `H = ∂r/∂δ`
/// (matching `iekf.rs`), so the −1 Jacobian entries pull travel toward 0 at a
/// topout.
///
/// # Panics
/// Does not panic. Degenerate or non-finite updates are silently skipped (the
/// pre-update belief is retained for that factor).
pub fn forward_filter(
    drive: &[f64],
    stationary: &[bool],
    airborne: &[bool],
    params: &WheelParams,
) -> Vec<FilteredSample> {
    let n = drive.len();
    let mut out = Vec::with_capacity(n);

    let mut w  = 0.0_f64;
    let mut wv = 0.0_f64;
    let mut p00 = params.init_travel_var;
    let mut p01 = 0.0_f64;
    let mut p11 = params.init_vel_var;

    let dt = params.dt;
    let r_zupt    = params.zupt_sigma    * params.zupt_sigma;
    let r_topout  = params.topout_sigma  * params.topout_sigma;
    let r_barrier = params.barrier_sigma * params.barrier_sigma;
    let r_sag     = params.sag_sigma     * params.sag_sigma;

    for i in 0..n {
        // ── 1. Predict ───────────────────────────────────────────────────────
        // Position uses OLD velocity (explicit Euler): w⁺ = w + wv·dt
        w  += wv * dt;
        wv += drive[i] * dt;
        let (pp00, pp01, pp11) = predict_cov(p00, p01, p11, dt, params.q_pos, params.q_vel);
        p00 = pp00;
        p01 = pp01;
        p11 = pp11;

        // ── 2. ZUPT (stationary) ──────────────────────────────────────────────
        if stationary[i] {
            // r = 0 − wv,  H = [0, −1]
            let r = -wv;
            scalar_update(&mut w, &mut wv, &mut p00, &mut p01, &mut p11,
                          0.0, -1.0, r, r_zupt);
        }

        // ── 3. Airborne / sag ─────────────────────────────────────────────────
        if airborne[i] {
            // Topout: r = 0 − w,  H = [−1, 0]
            let r = -w;
            scalar_update(&mut w, &mut wv, &mut p00, &mut p01, &mut p11,
                          -1.0, 0.0, r, r_topout);
            // Zero wheel velocity during airborne: r = 0 − wv,  H = [0, −1]
            let r = -wv;
            scalar_update(&mut w, &mut wv, &mut p00, &mut p01, &mut p11,
                          0.0, -1.0, r, r_zupt);
        } else if !stationary[i] && params.use_sag_prior {
            // Sag prior: r = sag − w,  H = [−1, 0]
            let r_sag_res = params.sag - w;
            scalar_update(&mut w, &mut wv, &mut p00, &mut p01, &mut p11,
                          -1.0, 0.0, r_sag_res, r_sag);
        }

        // ── 4. Barrier (joint 2-row, always attempted) ───────────────────────
        // Residuals mirror the engine's `TravelBarrier` exactly (prior.rs):
        // r = [relu(−w), relu(w − travel_max)], H rows [−1, 0] / [+1, 0] active
        // only outside the band.
        // Row 0: lower bound violation (w < 0)
        let (h00, h01, r0) = if w < 0.0 { (-1.0, 0.0, -w) } else { (0.0, 0.0, 0.0) };
        // Row 1: upper bound violation (w > travel_max)
        let (h10, h11, r1) = if w > params.travel_max {
            (1.0, 0.0, w - params.travel_max)
        } else {
            (0.0, 0.0, 0.0)
        };
        barrier_update(
            &mut w, &mut wv, &mut p00, &mut p01, &mut p11,
            h00, h01, h10, h11, r0, r1, r_barrier,
        );

        out.push(FilteredSample { w, wv, p00, p01, p11 });
    }

    out
}

/// RTS backward smoother over a previously computed [`forward_filter`] output.
///
/// At each step `k` from `n−2` down to 0 the smoother re-derives the predicted
/// belief the forward pass had at `k+1`, computes the RTS gain
/// `C = P_k · Fᵀ · P_pred⁻¹`, then fuses the backward-pass correction:
///
/// ```text
/// m_k^s = m_k  +  C · (m_{k+1}^s − m_pred)
/// P_k^s = P_k  +  C · (P_{k+1}^s − P_pred) · Cᵀ
/// ```
///
/// When `P_pred` is singular (det ≤ 1e-300) the gain is set to zero and the
/// smoothed estimate falls back to the filtered estimate for that step — this
/// prevents NaN propagation backward through the chain.
///
/// `drive[i]` is the same differential-acceleration drive used in the forward
/// pass, s.t. the predicted mean at `k+1` can be reconstructed as
/// `m_pred = F·m_k + [0, drive[k+1]·dt]`. Note that `drive[k+1]` corresponds to
/// the prediction step entering sample `k+1`.
pub fn rts_smooth(
    drive: &[f64],
    filtered: &[FilteredSample],
    params: &WheelParams,
) -> Vec<SmoothedSample> {
    let n = filtered.len();
    if n == 0 {
        return Vec::new();
    }
    let mut smoothed = vec![SmoothedSample { w: 0.0, wv: 0.0, p00: 0.0 }; n];

    // Initialise the last sample from the forward filter. `sp` carries the FULL
    // smoothed 2×2 covariance per step — the backward recursion needs all of it
    // at k+1, even though only p00 is exposed on [`SmoothedSample`].
    let last = &filtered[n - 1];
    smoothed[n - 1] = SmoothedSample { w: last.w, wv: last.wv, p00: last.p00 };
    let mut sp = (last.p00, last.p01, last.p11);

    let dt = params.dt;
    let q_pos = params.q_pos;
    let q_vel = params.q_vel;

    for k in (0..n - 1).rev() {
        let fk = &filtered[k];
        let m_k = [fk.w, fk.wv];
        let pk00 = fk.p00;
        let pk01 = fk.p01;
        let pk11 = fk.p11;

        // Re-derive the predicted moments the forward pass had at k+1.
        // m_pred = F · m_k + [0, drive[k+1]·dt]  (drive entry for the k+1 predict step)
        let m_pred_w = m_k[0] + m_k[1] * dt;
        let m_pred_wv = m_k[1] + drive[k + 1] * dt;
        let (pp00, pp01, pp11) = predict_cov(pk00, pk01, pk11, dt, q_pos, q_vel);

        // C = P_k · Fᵀ · P_pred⁻¹  (2×2 · 2×2 · 2×2)
        // P_k · Fᵀ: Fᵀ = [[1, 0], [dt, 1]]
        //   row 0: [pk00 + pk01·dt,  pk01]
        //   row 1: [pk01 + pk11·dt,  pk11]
        let pft00 = pk00 + pk01 * dt;
        let pft01 = pk01;
        let pft10 = pk01 + pk11 * dt;
        let pft11 = pk11;

        // Invert P_pred
        let (c00, c01, c10, c11) = match inv2(pp00, pp01, pp11) {
            None => {
                // Degenerate: smoothed = filtered for this step
                smoothed[k] = SmoothedSample { w: fk.w, wv: fk.wv, p00: fk.p00 };
                sp = (fk.p00, fk.p01, fk.p11);
                continue;
            }
            Some((si00, si01, si11)) => {
                // C = PFᵀ · P_pred⁻¹  (both 2×2)
                let c00 = pft00 * si00 + pft01 * si01;
                let c01 = pft00 * si01 + pft01 * si11;
                let c10 = pft10 * si00 + pft11 * si01;
                let c11 = pft10 * si01 + pft11 * si11;
                (c00, c01, c10, c11)
            }
        };

        // Innovation for the backward pass:
        let sk1 = &smoothed[k + 1];
        let dm_w = sk1.w - m_pred_w;
        let dm_wv = sk1.wv - m_pred_wv;

        // m_k^s = m_k + C · (m_{k+1}^s − m_pred)
        let sw = m_k[0] + c00 * dm_w + c01 * dm_wv;
        let swv = m_k[1] + c10 * dm_w + c11 * dm_wv;

        // P_k^s = P_k + C · (P_{k+1}^s − P_pred) · Cᵀ, with P_{k+1}^s the FULL
        // smoothed covariance carried in `sp` (approximating its off-diagonals
        // with filtered values would corrupt the recursion).
        let dp00 = sp.0 - pp00;
        let dp01 = sp.1 - pp01;
        let dp11 = sp.2 - pp11;

        // C · DP  (2×2 · symmetric 2×2)
        let (cdp00, cdp01, cdp10, cdp11) = mat2_mul_fp(c00, c01, c10, c11, dp00, dp01, dp11);
        // (C·DP)·Cᵀ with Cᵀ[j][k] = C[k][j] (C is NOT symmetric):
        //   [0,0] = cdp00·c00 + cdp01·c01,  [0,1] = cdp00·c10 + cdp01·c11,
        //   [1,1] = cdp10·c10 + cdp11·c11.
        let cdpct00 = cdp00 * c00 + cdp01 * c01;
        let cdpct01 = cdp00 * c10 + cdp01 * c11;
        let cdpct11 = cdp10 * c10 + cdp11 * c11;

        sp = (pk00 + cdpct00, pk01 + cdpct01, pk11 + cdpct11);
        smoothed[k] = SmoothedSample { w: sw, wv: swv, p00: sp.0 };
    }

    smoothed
}

/// Convenience wrapper: runs [`forward_filter`] then [`rts_smooth`] and returns
/// `(smoothed_travel_m, smoothed_velocity_m_s)`.
///
/// Returns `(vec![], vec![])` on empty input.
pub fn smooth_wheel(
    drive: &[f64],
    stationary: &[bool],
    airborne: &[bool],
    params: &WheelParams,
) -> (Vec<f64>, Vec<f64>) {
    if drive.is_empty() {
        return (vec![], vec![]);
    }
    let filtered = forward_filter(drive, stationary, airborne, params);
    let smoothed = rts_smooth(drive, &filtered, params);
    let w  = smoothed.iter().map(|s| s.w ).collect();
    let wv = smoothed.iter().map(|s| s.wv).collect();
    (w, wv)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference params used across tests (see task spec).
    fn ref_params(dt: f64) -> WheelParams {
        WheelParams {
            dt,
            q_pos:           (1e-3_f64).powi(2) * dt,
            q_vel:           5.0_f64.powi(2)    * dt,
            init_travel_var: (0.05_f64).powi(2),
            init_vel_var:    1.0_f64.powi(2),
            zupt_sigma:      0.02,
            topout_sigma:    0.01,
            barrier_sigma:   0.005,
            sag_sigma:       0.5,
            sag:             0.046,
            travel_max:      0.170,
            use_sag_prior:   false,
        }
    }

    // ── Test 1 ─────────────────────────────────────────────────────────────────
    #[test]
    fn forward_filter_static_input_flagged_stationary_travel_and_velocity_stay_at_zero() {
        // Arrange
        let dt = 0.01;
        let n  = 500;
        let drive: Vec<f64>    = vec![0.0; n];
        let stationary         = vec![true; n];
        let airborne           = vec![false; n];
        let params             = ref_params(dt);

        // Act
        let filtered = forward_filter(&drive, &stationary, &airborne, &params);

        // Assert — ZUPT every sample keeps both states at zero
        for s in &filtered {
            assert!(s.w.abs()  < 1e-6, "w={} expected ≈0", s.w);
            assert!(s.wv.abs() < 1e-6, "wv={} expected ≈0", s.wv);
        }
    }

    // ── Test 2 ─────────────────────────────────────────────────────────────────
    #[test]
    fn forward_filter_known_sinusoid_drive_tracks_travel_and_velocity() {
        // Arrange
        let dt  = 0.005_f64;
        let a   = 0.03_f64;           // amplitude, m
        let omega = std::f64::consts::PI; // rad/s
        let n   = (1.0 / dt).round() as usize + 1; // cover t in [0, 1.0]
        let drive: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 * dt;
                a * omega * omega * (omega * t).cos()
            })
            .collect();
        let stationary = vec![false; n];
        let airborne   = vec![false; n];
        let mut params = ref_params(dt);
        params.travel_max = 0.170;
        params.use_sag_prior = false;

        // Act
        let filtered = forward_filter(&drive, &stationary, &airborne, &params);

        // Assert — at t = 1.0 s (half period), truth: w = 2A = 0.06 m, ẇ = 0 m/s
        let idx_1s = (1.0_f64 / dt).round() as usize;
        let truth_w  = 2.0 * a;
        let truth_wv = 0.0;
        let s = filtered[idx_1s];
        assert!(
            (s.w - truth_w).abs() < 3e-3,
            "w at t=1s: got {:.6}, truth {:.6}, diff {:.2e}",
            s.w, truth_w, (s.w - truth_w).abs()
        );
        assert!(
            (s.wv - truth_wv).abs() < 0.03,
            "wv at t=1s: got {:.6}, truth {:.6}, diff {:.2e}",
            s.wv, truth_wv, (s.wv - truth_wv).abs()
        );
    }

    // ── Test 3 ─────────────────────────────────────────────────────────────────
    #[test]
    fn rts_smooth_biased_drive_with_terminal_topout_pre_topout_drift_reduced() {
        // Arrange
        let dt   = 0.005_f64;
        let n_st = 100;   // stationary phase
        let n_rd = 800;   // riding phase (biased drive accumulates error)
        let n_ab = 100;   // airborne phase (topout anchor)
        let n    = n_st + n_rd + n_ab;
        // Constant positive-acceleration drive — a pure DC error (truth w≡0). Small
        // enough that the drifted forward travel stays INSIDE [0, travel_max]
        // (peak ½·0.02·4² = 0.16 m < 0.17 m): with the barrier silent, the pre-topout
        // interval is anchored only at its two ends, which is exactly the boundary-
        // value structure the RTS pass exists to exploit.
        let drive: Vec<f64>  = vec![0.02; n];
        let mut stationary   = vec![false; n];
        let mut airborne     = vec![false; n];
        for i in 0..n_st { stationary[i] = true; }
        for i in (n_st + n_rd)..n { airborne[i] = true; }
        let mut params = ref_params(dt);
        params.use_sag_prior = false;

        // Act
        let filtered = forward_filter(&drive, &stationary, &airborne, &params);
        let smoothed = rts_smooth(&drive, &filtered, &params);

        // Assert — smoothed RMS over riding segment < 0.5 × forward RMS
        let riding = n_st..(n_st + n_rd);
        let fwd_rms: f64 = {
            let ss: f64 = riding.clone().map(|i| filtered[i].w.powi(2)).sum();
            (ss / n_rd as f64).sqrt()
        };
        let smt_rms: f64 = {
            let ss: f64 = riding.clone().map(|i| smoothed[i].w.powi(2)).sum();
            (ss / n_rd as f64).sqrt()
        };
        assert!(
            smt_rms < 0.5 * fwd_rms,
            "smoothed RMS {:.6} should be < 0.5 × forward RMS {:.6}",
            smt_rms, fwd_rms
        );
    }

    // ── Test 4 ─────────────────────────────────────────────────────────────────
    #[test]
    fn rts_smooth_smoothed_travel_variance_never_exceeds_filtered() {
        // Arrange  (same scenario as test 3)
        let dt   = 0.005_f64;
        let n_st = 100;
        let n_rd = 800;
        let n_ab = 100;
        let n    = n_st + n_rd + n_ab;
        let drive        = vec![0.5_f64; n];
        let mut stationary = vec![false; n];
        let mut airborne   = vec![false; n];
        for i in 0..n_st       { stationary[i] = true; }
        for i in (n_st + n_rd)..n { airborne[i] = true; }
        let params = ref_params(dt);

        // Act
        let filtered = forward_filter(&drive, &stationary, &airborne, &params);
        let smoothed = rts_smooth(&drive, &filtered, &params);

        // Assert — p00_smoothed ≤ p00_filtered + tolerance at every step
        for i in 0..n {
            assert!(
                smoothed[i].p00 <= filtered[i].p00 + 1e-12,
                "sample {i}: smoothed p00={:.3e} > filtered p00={:.3e}",
                smoothed[i].p00, filtered[i].p00
            );
        }
    }

    // ── Test 5 ─────────────────────────────────────────────────────────────────
    #[test]
    fn rts_smooth_no_active_factors_equals_the_forward_pass() {
        // Arrange — zero drive, no event flags, sag off.  With initial state
        // [0, 0] and drive ≡ 0 the mean stays at w=0, wv=0 throughout.  The
        // barrier check `w < 0` is false at w=0 (strict), so the barrier is a
        // true no-op.  No ZUPT, no topout, no sag prior fires.  Therefore no
        // factor injects future information and the backward pass must recover
        // the filtered means exactly — this pins the C-gain algebra.
        let dt     = 0.005_f64;
        let n      = 400;
        let drive  = vec![0.0_f64; n];
        let stationary = vec![false; n];
        let airborne   = vec![false; n];
        let mut params = ref_params(dt);
        params.use_sag_prior = false;
        params.travel_max    = 0.170;

        // Act
        let filtered = forward_filter(&drive, &stationary, &airborne, &params);
        let smoothed = rts_smooth(&drive, &filtered, &params);

        // Assert — backward correction is zero (no future information)
        for i in 0..n {
            assert!(
                (smoothed[i].w  - filtered[i].w ).abs() < 1e-9,
                "sample {i}: Δw = {:.3e}", smoothed[i].w - filtered[i].w
            );
            assert!(
                (smoothed[i].wv - filtered[i].wv).abs() < 1e-9,
                "sample {i}: Δwv = {:.3e}", smoothed[i].wv - filtered[i].wv
            );
        }
    }

    // ── Test 6 ─────────────────────────────────────────────────────────────────
    #[test]
    fn smooth_wheel_empty_input_returns_empty_vectors() {
        // Arrange
        let params = ref_params(0.005);

        // Act
        let (w, wv) = smooth_wheel(&[], &[], &[], &params);

        // Assert
        assert!(w.is_empty(),  "expected empty travel vec");
        assert!(wv.is_empty(), "expected empty velocity vec");
    }
}
