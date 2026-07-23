//! Backward pass for the 3DGUT (Unscented Transform) projection —
//! matches `calc_mean_cov2d_ut` in `brush_render::kernels::helpers`.
//!
//! Camera-model-agnostic, like the forward: reuses the existing
//! `project`/`calculate_project_jacobian` dispatch (evaluated at each of
//! the 7 sigma points instead of just the mean) rather than hand-deriving
//! a new per-model Hessian the way the affine path's
//! `calculate_projection_vjp_*` functions do.

use brush_cube::{
    UT_SIGMA_SCALE, UT_WEIGHT_COV_CENTER, UT_WEIGHT_MEAN_CENTER, UT_WEIGHT_OUTER, Vec2,
};
use brush_render::kernels::camera_model::{
    CameraModel, JacobianClampLimits, calculate_project_jacobian, project,
};
use brush_render::kernels::types::{Mat3, ProjectUniforms, Quat, Sym2, Vec3A};
use burn_cubecl::cubecl;
use burn_cubecl::cubecl::cube;
use burn_cubecl::cubecl::prelude::*;

/// Given upstream `v_mean2d`/`v_cov2d` (gradients w.r.t. the UT-reconstructed
/// mean2d/cov2d), returns `(v_mean_c, v_ns)`: the camera-space gradient
/// w.r.t. the splat's view-space mean, and the gradient w.r.t. the
/// view-space "square root" matrix `ns = view_rotation * quat_to_mat3 *
/// diag(scale)` (same `ns` the forward pass builds sigma points from).
///
/// `v_ns` feeds into the exact same object-space conversion + `v_scale`/
/// `v_quat` machinery the affine path already uses in
/// `project_backwards_kernel` — that code only cares about the final
/// linear-map gradient, not how it was produced.
#[cube]
pub fn calculate_ut_vjp(
    scale: Vec3A,
    quat: Quat,
    mean_c: Vec3A,
    u: ProjectUniforms,
    v_mean2d: Vec2,
    v_cov2d: Sym2,
    #[comptime] camera_model: CameraModel,
) -> (Vec3A, Mat3) {
    let ns = u.view_rotation().mul_mat3(quat.to_mat3()).mul_diag(scale);
    let c0 = ns.col0().scale(UT_SIGMA_SCALE);
    let c1 = ns.col1().scale(UT_SIGMA_SCALE);
    let c2 = ns.col2().scale(UT_SIGMA_SCALE);

    let p0 = mean_c;
    let p1 = mean_c.add(c0);
    let p2 = mean_c.add(c1);
    let p3 = mean_c.add(c2);
    let p4 = mean_c.sub(c0);
    let p5 = mean_c.sub(c1);
    let p6 = mean_c.sub(c2);

    let (u0x, u0y) = project(p0, u.pinhole_params, camera_model);
    let (u1x, u1y) = project(p1, u.pinhole_params, camera_model);
    let (u2x, u2y) = project(p2, u.pinhole_params, camera_model);
    let (u3x, u3y) = project(p3, u.pinhole_params, camera_model);
    let (u4x, u4y) = project(p4, u.pinhole_params, camera_model);
    let (u5x, u5y) = project(p5, u.pinhole_params, camera_model);
    let (u6x, u6y) = project(p6, u.pinhole_params, camera_model);

    let mean2d_x =
        UT_WEIGHT_MEAN_CENTER * u0x + UT_WEIGHT_OUTER * (u1x + u2x + u3x + u4x + u5x + u6x);
    let mean2d_y =
        UT_WEIGHT_MEAN_CENTER * u0y + UT_WEIGHT_OUTER * (u1y + u2y + u3y + u4y + u5y + u6y);

    let d0x = u0x - mean2d_x;
    let d0y = u0y - mean2d_y;
    let d1x = u1x - mean2d_x;
    let d1y = u1y - mean2d_y;
    let d2x = u2x - mean2d_x;
    let d2y = u2y - mean2d_y;
    let d3x = u3x - mean2d_x;
    let d3y = u3y - mean2d_y;
    let d4x = u4x - mean2d_x;
    let d4y = u4y - mean2d_y;
    let d5x = u5x - mean2d_x;
    let d5y = u5y - mean2d_y;
    let d6x = u6x - mean2d_x;
    let d6y = u6y - mean2d_y;

    let v00 = v_cov2d.c00;
    let v01 = v_cov2d.c01;
    let v11 = v_cov2d.c11;

    // Raw per-point covariance-path gradient, treating each d_i as if it
    // were an independent variable (the `mean2d` cross term is added back
    // in below via `corr`). `v_cov2d.c01` comes out of `inverse2x2_vjp`
    // as a single-slot matrix gradient (matching how the rest of the
    // codebase consumes `Sym2` as a full symmetric matrix via congruence
    // transforms) — since `cov2d = Σ wc_i d_i ⊗ d_i` contributes to
    // *both* the (0,1) and (1,0) slots identically, the off-diagonal
    // term needs the same `2×` factor the diagonal terms already have.
    let g0x = UT_WEIGHT_COV_CENTER * (2.0f32 * v00 * d0x + 2.0f32 * v01 * d0y);
    let g0y = UT_WEIGHT_COV_CENTER * (2.0f32 * v11 * d0y + 2.0f32 * v01 * d0x);
    let g1x = UT_WEIGHT_OUTER * (2.0f32 * v00 * d1x + 2.0f32 * v01 * d1y);
    let g1y = UT_WEIGHT_OUTER * (2.0f32 * v11 * d1y + 2.0f32 * v01 * d1x);
    let g2x = UT_WEIGHT_OUTER * (2.0f32 * v00 * d2x + 2.0f32 * v01 * d2y);
    let g2y = UT_WEIGHT_OUTER * (2.0f32 * v11 * d2y + 2.0f32 * v01 * d2x);
    let g3x = UT_WEIGHT_OUTER * (2.0f32 * v00 * d3x + 2.0f32 * v01 * d3y);
    let g3y = UT_WEIGHT_OUTER * (2.0f32 * v11 * d3y + 2.0f32 * v01 * d3x);
    let g4x = UT_WEIGHT_OUTER * (2.0f32 * v00 * d4x + 2.0f32 * v01 * d4y);
    let g4y = UT_WEIGHT_OUTER * (2.0f32 * v11 * d4y + 2.0f32 * v01 * d4x);
    let g5x = UT_WEIGHT_OUTER * (2.0f32 * v00 * d5x + 2.0f32 * v01 * d5y);
    let g5y = UT_WEIGHT_OUTER * (2.0f32 * v11 * d5y + 2.0f32 * v01 * d5x);
    let g6x = UT_WEIGHT_OUTER * (2.0f32 * v00 * d6x + 2.0f32 * v01 * d6y);
    let g6y = UT_WEIGHT_OUTER * (2.0f32 * v11 * d6y + 2.0f32 * v01 * d6x);

    let g_sum_x = g0x + g1x + g2x + g3x + g4x + g5x + g6x;
    let g_sum_y = g0y + g1y + g2y + g3y + g4y + g5y + g6y;

    // `d_i = proj_i - mean2d` depends on every `proj_k` through `mean2d`,
    // so `v_proj_k = g_k + w_mean_k * (v_mean2d - sum_i g_i)`.
    let corr_x = v_mean2d.x() - g_sum_x;
    let corr_y = v_mean2d.y() - g_sum_y;

    let vp0 = Vec2::new(
        g0x + UT_WEIGHT_MEAN_CENTER * corr_x,
        g0y + UT_WEIGHT_MEAN_CENTER * corr_y,
    );
    let vp1 = Vec2::new(g1x + UT_WEIGHT_OUTER * corr_x, g1y + UT_WEIGHT_OUTER * corr_y);
    let vp2 = Vec2::new(g2x + UT_WEIGHT_OUTER * corr_x, g2y + UT_WEIGHT_OUTER * corr_y);
    let vp3 = Vec2::new(g3x + UT_WEIGHT_OUTER * corr_x, g3y + UT_WEIGHT_OUTER * corr_y);
    let vp4 = Vec2::new(g4x + UT_WEIGHT_OUTER * corr_x, g4y + UT_WEIGHT_OUTER * corr_y);
    let vp5 = Vec2::new(g5x + UT_WEIGHT_OUTER * corr_x, g5y + UT_WEIGHT_OUTER * corr_y);
    let vp6 = Vec2::new(g6x + UT_WEIGHT_OUTER * corr_x, g6y + UT_WEIGHT_OUTER * corr_y);

    // `u.jacobian_clamp_limits` bounds the Pinhole/RT8 Jacobian's `x/z`,
    // `y/z` terms so a *single* evaluation at the mean (always inside the
    // frustum, by construction) doesn't blow up near the FOV edge. UT
    // evaluates the Jacobian at 7 different points, several of which sit
    // well outside that "trusted" region for large/tilted splats — and
    // the forward pass they must match (`calc_mean_cov2d_ut`) never
    // clamps anything (it only ever calls the exact `project`). Reusing
    // the mean-tuned clamp here would silently desync backward from
    // forward, so every per-sigma-point Jacobian below is evaluated
    // wide-open instead.
    let wide_clamp = JacobianClampLimits {
        lim_pos_x: 1.0e9f32,
        lim_pos_y: 1.0e9f32,
        lim_neg_x: -1.0e9f32,
        lim_neg_y: -1.0e9f32,
    };
    let j0 = calculate_project_jacobian(p0, wide_clamp, u.pinhole_params, camera_model);
    let j1 = calculate_project_jacobian(p1, wide_clamp, u.pinhole_params, camera_model);
    let j2 = calculate_project_jacobian(p2, wide_clamp, u.pinhole_params, camera_model);
    let j3 = calculate_project_jacobian(p3, wide_clamp, u.pinhole_params, camera_model);
    let j4 = calculate_project_jacobian(p4, wide_clamp, u.pinhole_params, camera_model);
    let j5 = calculate_project_jacobian(p5, wide_clamp, u.pinhole_params, camera_model);
    let j6 = calculate_project_jacobian(p6, wide_clamp, u.pinhole_params, camera_model);

    let vpt0 = j0.transpose_mul_vec2(vp0);
    let vpt1 = j1.transpose_mul_vec2(vp1);
    let vpt2 = j2.transpose_mul_vec2(vp2);
    let vpt3 = j3.transpose_mul_vec2(vp3);
    let vpt4 = j4.transpose_mul_vec2(vp4);
    let vpt5 = j5.transpose_mul_vec2(vp5);
    let vpt6 = j6.transpose_mul_vec2(vp6);

    let v_mean_c = vpt0
        .add(vpt1)
        .add(vpt2)
        .add(vpt3)
        .add(vpt4)
        .add(vpt5)
        .add(vpt6);

    // p{1,2,3} = mean_c + c*ns_col{0,1,2}; p{4,5,6} = mean_c - c*ns_col{0,1,2}.
    let v_ns_col0 = vpt1.sub(vpt4).scale(UT_SIGMA_SCALE);
    let v_ns_col1 = vpt2.sub(vpt5).scale(UT_SIGMA_SCALE);
    let v_ns_col2 = vpt3.sub(vpt6).scale(UT_SIGMA_SCALE);

    (v_mean_c, Mat3::from_cols(v_ns_col0, v_ns_col1, v_ns_col2))
}
