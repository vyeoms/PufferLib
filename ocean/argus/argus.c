#include "argus.h"
#include <stdio.h>

int main(void) {
    load_argus_urdf(&g_argus_robot, "resources/argus/argus_dof20_minimum.urdf");
    argus_robot_defaults(&g_argus_robot);
    argus_robot_compute(&g_argus_robot);
    argus_compute_obs_dims(&g_argus_robot);

    int n_dof = g_argus_robot.n_dof;
    float total_mass = g_argus_robot.body_mass;
    for (int i = 0; i < n_dof; i++) total_mass += g_argus_robot.leg_mass[i];
    printf("Robot: %d DOF, total_mass=%.1f kg\n", n_dof, total_mass);

    Argus env = {0};
    env.num_agents = 1;
    env.terrain_height    = 0.12f;
    env.terrain_min_size  = 0.4f;
    env.terrain_max_size  = 2.0f;
    env.terrain_num_rects = 80;
    env.terrain_platform  = 2.0f;

    env.observations = (float*)calloc(g_obs_total, sizeof(float));
    env.actions      = (float*)calloc(n_dof, sizeof(float));
    env.rewards      = (float*)calloc(1, sizeof(float));
    env.terminals    = (float*)calloc(1, sizeof(float));

    c_reset(&env);
    c_render(&env);

    int step = 0;
    while (!WindowShouldClose()) {
        for (int i = 0; i < n_dof; i++)
            env.actions[i] = (float)rand_r(&env.rng) / (float)RAND_MAX * 2.0f - 1.0f;

        c_step(&env);
        c_render(&env);
        step++;

        if (step % 100 == 0) {
            float t = fmaxf(1.0f, (float)env.tick);
            printf("step %d  pos=(%.3f,%.3f,%.3f)  rew=%.3f  "
                   "lv=%.3f df=%.3f dv=%.3f dl=%.4f im=%.4f sl=%.4f "
                   "mu=%.2f dm=%.1f\n",
                   step, env.pos.x, env.pos.y, env.pos.z,
                   env.rewards[0],
                   env.ep_rew_lin_vel/t, env.ep_rew_dof_force/t,
                   env.ep_rew_dof_vel/t, env.ep_rew_dof_limit/t,
                   env.ep_rew_impact/t, env.ep_rew_slip/t,
                   env.mu_friction, env.mass_offset);
        }
    }

    free(env.observations);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
    c_close(&env);
    return 0;
}
