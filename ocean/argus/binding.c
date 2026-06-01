#include "argus.h"

#define OBS_SIZE    (N_STACK * (3 + 2 + 20*3 + 9))
#define NUM_ATNS    20
#define ACT_SIZES   {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1}
#define OBS_TENSOR_T FloatTensor

#define Env Argus
#include "vecenv.h"

static float env_get(Dict* d, const char* key, float fallback) {
    DictItem* item = dict_get_unsafe(d, key);
    return item ? (float)item->value : fallback;
}

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    if (!g_argus_robot.loaded) {
        load_argus_urdf(&g_argus_robot, "resources/argus/argus_dof20_minimum.urdf");
        argus_robot_defaults(&g_argus_robot);
        argus_robot_compute(&g_argus_robot);
        argus_compute_obs_dims(&g_argus_robot);
    }
    env->terrain_height    = env_get(kwargs, "terrain_height",    0.12f);
    env->terrain_min_size  = env_get(kwargs, "terrain_min_size",  0.4f);
    env->terrain_max_size  = env_get(kwargs, "terrain_max_size",  2.0f);
    env->terrain_num_rects = (int)env_get(kwargs, "terrain_num_rects", 80.0f);
    env->terrain_platform  = env_get(kwargs, "terrain_platform",  2.0f);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "episode_return",  log->episode_return);
    dict_set(out, "episode_length",  log->episode_length);
    dict_set(out, "mean_vel_error",  log->mean_vel_error);
    dict_set(out, "score",           log->score);
    dict_set(out, "perf",            log->perf);
    dict_set(out, "rew_lin_vel",     log->rew_lin_vel);
    dict_set(out, "rew_dof_force",   log->rew_dof_force);
    dict_set(out, "rew_dof_vel",     log->rew_dof_vel);
    dict_set(out, "rew_dof_limit",   log->rew_dof_limit);
    dict_set(out, "rew_impact",      log->rew_impact);
    dict_set(out, "rew_slip",        log->rew_slip);
    dict_set(out, "rew_action",      log->rew_action);
    dict_set(out, "rew_action_rate", log->rew_action_rate);
}
