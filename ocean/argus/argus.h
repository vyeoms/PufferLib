#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "raylib.h"

#define MAX_DOF           32
#define N_STACK            3

// Heightfield terrain
#define HF_RES           128
#define HF_SCALE         0.08f

#define GRAVITY         9.81f
#define ACTION_SCALE    0.25f

// PD+I controller (from argus.yaml)
#define KP              2000.0f
#define KD               100.0f
#define KI               200.0f
#define JOINT_DAMP        10.0f

// robstride-02 motor model
#define V_MOTOR         48.0f
#define V_BACK_EMF       6.0f
#define R_WHEEL          0.04f
#define T_MAX           15.0f
#define F_MAX_ABS      (T_MAX / R_WHEEL)

#define K_CONTACT    30000.0f
#define B_CONTACT      300.0f
#define MAX_PEN        0.01f

// Timing (from argus.yaml)
#define DT_SIM          0.005f
#define DECIMATION         4
#define DT_ACT          (DT_SIM * DECIMATION)

// Velocity low-pass filter
#define DQ_ALPHA        0.1f

// Episode settings
#define MAX_STEPS       1000
#define CMD_VEL_MAX     1.0f
#define CMD_RESAMPLE    0.005f

// Observation scaling (from argus.yaml)
#define LIN_VEL_SCALE   2.0f
#define ANG_VEL_SCALE   0.25f
#define DOF_POS_SCALE   1.0f
#define DOF_VEL_SCALE   0.05f

typedef struct {
    float episode_return;
    float episode_length;
    float mean_vel_error;
    float score;
    float perf;
    float rew_lin_vel;
    float rew_dof_force;
    float rew_dof_vel;
    float rew_dof_limit;
    float rew_impact;
    float rew_slip;
    float rew_action;
    float rew_action_rate;
    float n;
} Log;

typedef struct { float x, y, z; }       Vec3;
typedef struct { float w, x, y, z; }    Quat;  // scalar-first

static inline Vec3 v3add(Vec3 a, Vec3 b)  { return (Vec3){a.x+b.x, a.y+b.y, a.z+b.z}; }
static inline Vec3 v3scale(Vec3 a, float s){ return (Vec3){a.x*s, a.y*s, a.z*s}; }
static inline float v3dot(Vec3 a, Vec3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
static inline float v3norm(Vec3 a)        { return sqrtf(v3dot(a,a)); }
static inline Vec3 v3cross(Vec3 a, Vec3 b) {
    return (Vec3){a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x};
}
static inline float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}
static inline float fmaxf_(float a, float b) { return a > b ? a : b; }
static inline float fminf_(float a, float b) { return a < b ? a : b; }
static inline float fabsf_(float x) { return x < 0 ? -x : x; }

static inline Quat qmul(Quat p, Quat q) {
    return (Quat){
        p.w*q.w - p.x*q.x - p.y*q.y - p.z*q.z,
        p.w*q.x + p.x*q.w + p.y*q.z - p.z*q.y,
        p.w*q.y - p.x*q.z + p.y*q.w + p.z*q.x,
        p.w*q.z + p.x*q.y - p.y*q.x + p.z*q.w
    };
}

static inline Vec3 qrot(Quat q, Vec3 v) {
    float tx = 2.0f*(q.y*v.z - q.z*v.y);
    float ty = 2.0f*(q.z*v.x - q.x*v.z);
    float tz = 2.0f*(q.x*v.y - q.y*v.x);
    return (Vec3){
        v.x + q.w*tx + q.y*tz - q.z*ty,
        v.y + q.w*ty + q.z*tx - q.x*tz,
        v.z + q.w*tz + q.x*ty - q.y*tx
    };
}

static inline void qnorm(Quat *q) {
    float n = sqrtf(q->w*q->w + q->x*q->x + q->y*q->y + q->z*q->z);
    if (n > 1e-8f) { float inv=1.0f/n; q->w*=inv; q->x*=inv; q->y*=inv; q->z*=inv; }
}

static inline void qintegrate(Quat *q, Vec3 om, float dt) {
    Quat oq = {0.0f, om.x*0.5f, om.y*0.5f, om.z*0.5f};
    Quat dq = qmul(oq, *q);
    q->w += dq.w*dt; q->x += dq.x*dt; q->y += dq.y*dt; q->z += dq.z*dt;
    qnorm(q);
}

static inline void qmat(Quat q, float m[9]) {
    float x2=q.x*q.x, y2=q.y*q.y, z2=q.z*q.z;
    float xy=q.x*q.y, xz=q.x*q.z, yz=q.y*q.z;
    float wx=q.w*q.x, wy=q.w*q.y, wz=q.w*q.z;
    m[0]=1-2*(y2+z2); m[1]=2*(xy-wz);   m[2]=2*(xz+wy);
    m[3]=2*(xy+wz);   m[4]=1-2*(x2+z2); m[5]=2*(yz-wx);
    m[6]=2*(xz-wy);   m[7]=2*(yz+wx);   m[8]=1-2*(x2+y2);
}

static inline Quat qslerp(float t, Quat q1, Quat q2) {
    float dot = q1.w*q2.w + q1.x*q2.x + q1.y*q2.y + q1.z*q2.z;
    if (dot < 0.0f) {
        q2.w=-q2.w; q2.x=-q2.x; q2.y=-q2.y; q2.z=-q2.z;
        dot = -dot;
    }
    dot = clampf(dot, -1.0f, 1.0f);
    if (dot > 1.0f - 1e-6f) {
        Quat r = {q1.w + t*(q2.w-q1.w), q1.x + t*(q2.x-q1.x),
                  q1.y + t*(q2.y-q1.y), q1.z + t*(q2.z-q1.z)};
        qnorm(&r);
        return r;
    }
    float omega = acosf(dot);
    float inv_sin = 1.0f / sinf(omega);
    float s0 = sinf((1.0f - t) * omega) * inv_sin;
    float s1 = sinf(t * omega) * inv_sin;
    return (Quat){s0*q1.w + s1*q2.w, s0*q1.x + s1*q2.x,
                  s0*q1.y + s1*q2.y, s0*q1.z + s1*q2.z};
}

typedef struct {
    int    n_dof;
    float  body_mass;
    float  body_ixx, body_iyy, body_izz;
    float  sphere_radius;

    float  leg_mass[MAX_DOF];
    float  leg_dir[MAX_DOF][3];       // unit direction from body center
    float  leg_base_dist[MAX_DOF];    // distance from body center to joint origin
    float  joint_lo[MAX_DOF];
    float  joint_hi[MAX_DOF];
    float  joint_effort[MAX_DOF];
    float  foot_radius[MAX_DOF];

    int    loaded;
} ArgusRobot;

static ArgusRobot g_argus_robot = {0};

static float parse_attr_float(const char* tag, const char* attr) {
    char pattern[64];
    snprintf(pattern, sizeof(pattern), "%s=\"", attr);
    const char* p = strstr(tag, pattern);
    if (!p) return 0.0f;
    p += strlen(pattern);
    return (float)atof(p);
}

static void parse_xyz(const char* tag, float out[3]) {
    const char* p = strstr(tag, "xyz=\"");
    if (!p) { out[0]=out[1]=out[2]=0; return; }
    p += 5;
    out[0] = (float)strtod(p, (char**)&p);
    out[1] = (float)strtod(p, (char**)&p);
    out[2] = (float)strtod(p, (char**)&p);
}

static const char* next_tag(const char* p, const char* tagname) {
    char pattern[64];
    snprintf(pattern, sizeof(pattern), "<%s ", tagname);
    const char* found = strstr(p, pattern);
    if (found) return found;
    snprintf(pattern, sizeof(pattern), "<%s>", tagname);
    return strstr(p, pattern);
}

static const char* tag_end(const char* p) {
    while (*p && *p != '>') p++;
    return (*p == '>') ? p + 1 : p;
}

static void extract_tag_content(const char* start, const char* end, char* buf, int bufsize) {
    int len = (int)(end - start);
    if (len >= bufsize) len = bufsize - 1;
    memcpy(buf, start, len);
    buf[len] = '\0';
}

static void load_argus_urdf(ArgusRobot* robot, const char* path) {
    FILE* f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "argus: cannot open URDF '%s', using defaults\n", path);
        return;
    }

    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* xml = (char*)malloc(fsize + 1);
    if (!xml) { fclose(f); return; }
    fread(xml, 1, fsize, f);
    xml[fsize] = '\0';
    fclose(f);

    const char* base_link = strstr(xml, "name=\"base_link\"");
    if (base_link) {
        const char* inertial = strstr(base_link, "<inertial");
        if (inertial) {
            const char* mass_tag = next_tag(inertial, "mass");
            if (mass_tag) {
                char buf[256];
                extract_tag_content(mass_tag, tag_end(mass_tag), buf, sizeof(buf));
                robot->body_mass = parse_attr_float(buf, "value");
            }
            const char* inertia_tag = next_tag(inertial, "inertia");
            if (inertia_tag) {
                char buf[512];
                extract_tag_content(inertia_tag, tag_end(inertia_tag), buf, sizeof(buf));
                robot->body_ixx = parse_attr_float(buf, "ixx");
                robot->body_iyy = parse_attr_float(buf, "iyy");
                robot->body_izz = parse_attr_float(buf, "izz");
            }
        }
        const char* base_end = strstr(base_link, "</link");
        if (!base_end) base_end = xml + fsize;
        const char* collision = strstr(base_link, "<collision");
        if (collision && collision < base_end) {
            const char* sphere = strstr(collision, "<sphere");
            if (sphere && sphere < base_end) {
                char buf[256];
                extract_tag_content(sphere, tag_end(sphere), buf, sizeof(buf));
                robot->sphere_radius = parse_attr_float(buf, "radius");
            }
        }
        if (robot->sphere_radius <= 0.0f) {
            const char* vis = strstr(base_link, "<visual");
            if (vis && vis < base_end) {
                const char* sphere = strstr(vis, "<sphere");
                if (sphere && sphere < base_end) {
                    char buf[256];
                    extract_tag_content(sphere, tag_end(sphere), buf, sizeof(buf));
                    robot->sphere_radius = parse_attr_float(buf, "radius");
                }
            }
        }
    }

    if (robot->sphere_radius <= 0.0f) robot->sphere_radius = 0.25f;

    int n = 0;
    const char* cursor = xml;
    while (n < MAX_DOF) {
        const char* joint = next_tag(cursor, "joint");
        if (!joint) break;

        const char* jend = tag_end(joint);
        char jbuf[1024];
        extract_tag_content(joint, jend, jbuf, sizeof(jbuf));

        if (!strstr(jbuf, "prismatic")) {
            cursor = jend;
            continue;
        }

        const char* origin = strstr(jend, "<origin");
        if (!origin) { cursor = jend; continue; }

        const char* next_joint = next_tag(jend, "joint");

        char obuf[512];
        extract_tag_content(origin, tag_end(origin), obuf, sizeof(obuf));

        float xyz[3];
        parse_xyz(obuf, xyz);

        float dist = sqrtf(xyz[0]*xyz[0] + xyz[1]*xyz[1] + xyz[2]*xyz[2]);
        if (dist > 1e-6f) {
            robot->leg_dir[n][0] = xyz[0] / dist;
            robot->leg_dir[n][1] = xyz[1] / dist;
            robot->leg_dir[n][2] = xyz[2] / dist;
        }
        robot->leg_base_dist[n] = dist;

        const char* limit = strstr(jend, "<limit");
        if (limit && (!next_joint || limit < next_joint)) {
            char lbuf[512];
            extract_tag_content(limit, tag_end(limit), lbuf, sizeof(lbuf));
            robot->joint_lo[n] = parse_attr_float(lbuf, "lower");
            robot->joint_hi[n] = parse_attr_float(lbuf, "upper");
            robot->joint_effort[n] = parse_attr_float(lbuf, "effort");
        }

        const char* child = strstr(jend, "<link");
        if (child) {
            const char* child_end = strstr(child + 1, "</link");
            if (!child_end) child_end = xml + fsize;

            const char* child_inertial = strstr(child, "<inertial");
            if (child_inertial && child_inertial < child_end) {
                const char* child_mass = next_tag(child_inertial, "mass");
                if (child_mass && child_mass < child_end) {
                    char mbuf[256];
                    extract_tag_content(child_mass, tag_end(child_mass), mbuf, sizeof(mbuf));
                    robot->leg_mass[n] = parse_attr_float(mbuf, "value");
                }
            }

            const char* child_col = strstr(child, "<collision");
            if (child_col && child_col < child_end) {
                const char* child_sphere = strstr(child_col, "<sphere");
                if (child_sphere && child_sphere < child_end) {
                    char sbuf[256];
                    extract_tag_content(child_sphere, tag_end(child_sphere), sbuf, sizeof(sbuf));
                    robot->foot_radius[n] = parse_attr_float(sbuf, "radius");
                }
            }
            cursor = child_end;
        } else {
            cursor = jend;
        }

        n++;
    }

    robot->n_dof = n;
    robot->loaded = 1;
    free(xml);

    printf("argus: loaded URDF '%s': %d DOF, body_mass=%.1f, sphere_r=%.3f, "
           "ixx=%.2f iyy=%.2f izz=%.2f\n",
           path, n, robot->body_mass, robot->sphere_radius,
           robot->body_ixx, robot->body_iyy, robot->body_izz);
}

static void argus_robot_defaults(ArgusRobot* robot) {
    if (robot->loaded) return;

    #define LEG_S   0.5773503f
    #define LEG_P   0.9341724f
    #define LEG_Q   0.3568221f
    static const float DEFAULT_DIR[20][3] = {
        {-LEG_S,  LEG_S,  LEG_S}, { 0,      LEG_P,  LEG_Q},
        { 0,      LEG_P, -LEG_Q}, {-LEG_S,  LEG_S, -LEG_S},
        {-LEG_P,  LEG_Q,  0    }, { LEG_S,  LEG_S,  LEG_S},
        {-LEG_Q,  0,      LEG_P}, {-LEG_P, -LEG_Q,  0    },
        {-LEG_Q,  0,     -LEG_P}, { LEG_S,  LEG_S, -LEG_S},
        { LEG_S, -LEG_S,  LEG_S}, { 0,     -LEG_P,  LEG_Q},
        { 0,     -LEG_P, -LEG_Q}, { LEG_S, -LEG_S, -LEG_S},
        { LEG_P, -LEG_Q,  0    }, { LEG_Q,  0,      LEG_P},
        {-LEG_S, -LEG_S,  LEG_S}, {-LEG_S, -LEG_S, -LEG_S},
        { LEG_Q,  0,     -LEG_P}, { LEG_P,  LEG_Q,  0    },
    };
    #undef LEG_S
    #undef LEG_P
    #undef LEG_Q

    robot->n_dof = 20;
    robot->body_mass = 17.0f;
    robot->body_ixx = 0.9f;
    robot->body_iyy = 0.9f;
    robot->body_izz = 0.95f;
    robot->sphere_radius = 0.25f;

    for (int i = 0; i < 20; i++) {
        robot->leg_mass[i] = 0.3f;
        robot->leg_dir[i][0] = DEFAULT_DIR[i][0];
        robot->leg_dir[i][1] = DEFAULT_DIR[i][1];
        robot->leg_dir[i][2] = DEFAULT_DIR[i][2];
        robot->leg_base_dist[i] = robot->sphere_radius;
        robot->joint_lo[i] = -0.105f;
        robot->joint_hi[i] = 0.105f;
        robot->joint_effort[i] = 375.0f;
        robot->foot_radius[i] = 0.06f;
    }
    robot->loaded = 1;
}

static void argus_robot_compute(ArgusRobot* robot) {
    for (int i = 0; i < robot->n_dof; i++) {
        if (robot->foot_radius[i] <= 0.0f)
            robot->foot_radius[i] = 0.06f;
    }
}

static int g_obs_single_frame = 0;
static int g_obs_total = 0;

static void argus_compute_obs_dims(const ArgusRobot* robot) {
    g_obs_single_frame = 3 + 2 + robot->n_dof + robot->n_dof + robot->n_dof + 9;
    g_obs_total = g_obs_single_frame * N_STACK;
}

typedef struct {
    Log   log;
    float *observations;
    float *actions;
    float *rewards;
    float *terminals;
    int    num_agents;

    Vec3  pos;
    Quat  quat;
    Quat  quat_filt;
    Vec3  vel;
    Vec3  omega;

    float q[MAX_DOF];
    float dq[MAX_DOF];
    float dq_comp[MAX_DOF];
    float int_err[MAX_DOF];

    float cmd_vx, cmd_vy;
    float last_act[MAX_DOF];

    float obs_stack[N_STACK][3 + 2 + MAX_DOF*3 + 9];
    int   stack_head;

    float ep_return;
    float ep_vel_err_sum;
    int   tick;
    unsigned int rng;

    float mu_friction;
    float mass_offset;
    float total_mass_local;
    Vec3  com_offset;
    float kp_scale[MAX_DOF];
    float kd_scale[MAX_DOF];
    float dof_strength[MAX_DOF];
    float leg_mass_scale[MAX_DOF];
    float body_inertia_scale[3];
    float ori_delay;
    float action_delay;

    float action_to_use[MAX_DOF];

    float foot_contact_fz[MAX_DOF];
    float foot_horiz_vel_sq[MAX_DOF];
    int   foot_contact_count[MAX_DOF];

    float force_target[MAX_DOF];

    int   push_timer;

    Vec3  force_noise_vel;
    Vec3  force_noise_omega;

    float ep_rew_lin_vel;
    float ep_rew_dof_force;
    float ep_rew_dof_vel;
    float ep_rew_dof_limit;
    float ep_rew_impact;
    float ep_rew_slip;
    float ep_rew_action;
    float ep_rew_action_rate;

    float terrain_height;
    float terrain_min_size;
    float terrain_max_size;
    int   terrain_num_rects;
    float terrain_platform;

    float q_visual[MAX_DOF];

    float heightfield[HF_RES][HF_RES];
    float hf_origin_x, hf_origin_y;
} Argus;

static inline float randf_(unsigned int *rng) {
    return (float)rand_r(rng) / (float)RAND_MAX;
}
static inline float rand_range_(float lo, float hi, unsigned int *rng) {
    return lo + randf_(rng) * (hi - lo);
}
static inline float rand_log_uniform_(float lo, float hi, unsigned int *rng) {
    return expf(rand_range_(logf(lo), logf(hi), rng));
}
static inline Quat rand_quat_(unsigned int *rng) {
    float u1 = randf_(rng);
    float u2 = randf_(rng) * 2.0f * (float)M_PI;
    float u3 = randf_(rng) * 2.0f * (float)M_PI;
    float s1 = sqrtf(1.0f - u1), s2 = sqrtf(u1);
    return (Quat){s1 * sinf(u2), s1 * cosf(u2), s2 * sinf(u3), s2 * cosf(u3)};
}

static void generate_heightfield(Argus *env) {
    memset(env->heightfield, 0, sizeof(env->heightfield));
    env->hf_origin_x = env->pos.x;
    env->hf_origin_y = env->pos.y;

    float mh = env->terrain_height;
    if (mh <= 0.0f) return;

    float height_choices[4] = { -mh, -mh * 0.5f, mh * 0.5f, mh };
    int min_cells = (int)(env->terrain_min_size / HF_SCALE);
    int max_cells = (int)(env->terrain_max_size / HF_SCALE);
    if (min_cells < 1) min_cells = 1;
    if (max_cells < min_cells) max_cells = min_cells;

    for (int r = 0; r < env->terrain_num_rects; r++) {
        int w = min_cells + (int)(randf_(&env->rng) * (max_cells - min_cells));
        int h = min_cells + (int)(randf_(&env->rng) * (max_cells - min_cells));
        int si = (int)(randf_(&env->rng) * (HF_RES - w));
        int sj = (int)(randf_(&env->rng) * (HF_RES - h));
        float ht = height_choices[(int)(randf_(&env->rng) * 4) % 4];
        for (int i = si; i < si + w && i < HF_RES; i++)
            for (int j = sj; j < sj + h && j < HF_RES; j++)
                env->heightfield[i][j] = ht;
    }

    int plat = (int)(env->terrain_platform / HF_SCALE * 0.5f);
    int c = HF_RES / 2;
    for (int i = c - plat; i <= c + plat; i++)
        for (int j = c - plat; j <= c + plat; j++)
            if (i >= 0 && i < HF_RES && j >= 0 && j < HF_RES)
                env->heightfield[i][j] = 0.0f;
}

static float sample_heightfield(const Argus *env, float wx, float wy) {
    float lx = (wx - env->hf_origin_x) / HF_SCALE + HF_RES * 0.5f;
    float ly = (wy - env->hf_origin_y) / HF_SCALE + HF_RES * 0.5f;
    int ix = (int)lx;
    int iy = (int)ly;
    if (ix < 0 || ix >= HF_RES - 1 || iy < 0 || iy >= HF_RES - 1)
        return 0.0f;
    float fx = lx - ix;
    float fy = ly - iy;
    float h00 = env->heightfield[ix][iy];
    float h10 = env->heightfield[ix+1][iy];
    float h01 = env->heightfield[ix][iy+1];
    float h11 = env->heightfield[ix+1][iy+1];
    return h00*(1-fx)*(1-fy) + h10*fx*(1-fy) + h01*(1-fx)*fy + h11*fx*fy;
}

void build_frame(const Argus *env, float *dst) {
    const ArgusRobot *r = &g_argus_robot;
    int k = 0;
    dst[k++] = env->omega.x * ANG_VEL_SCALE;
    dst[k++] = env->omega.y * ANG_VEL_SCALE;
    dst[k++] = env->omega.z * ANG_VEL_SCALE;
    dst[k++] = env->cmd_vx * LIN_VEL_SCALE;
    dst[k++] = env->cmd_vy * LIN_VEL_SCALE;
    for (int i = 0; i < r->n_dof; i++) dst[k++] = env->q[i] * DOF_POS_SCALE;
    for (int i = 0; i < r->n_dof; i++) dst[k++] = env->dq_comp[i] * DOF_VEL_SCALE;
    for (int i = 0; i < r->n_dof; i++) dst[k++] = env->last_act[i];
    float mat[9];
    qmat(env->quat_filt, mat);
    for (int i = 0; i < 9; i++) dst[k++] = mat[i];
}

void flush_obs(Argus *env) {
    int frame_sz = g_obs_single_frame;
    for (int s = 0; s < N_STACK; s++) {
        int idx = (env->stack_head + s) % N_STACK;
        float *dst = env->observations + s * frame_sz;
        const float *src = env->obs_stack[idx];
        for (int k = 0; k < frame_sz; k++) dst[k] = src[k];
    }
}

void c_reset(Argus *env) {
    const ArgusRobot *r = &g_argus_robot;

    env->mu_friction = rand_range_(0.1f, 0.5f, &env->rng);
    env->mass_offset = rand_range_(-0.5f, 0.5f, &env->rng);
    env->com_offset = (Vec3){
        rand_range_(-0.05f, 0.05f, &env->rng),
        rand_range_(-0.05f, 0.05f, &env->rng),
        rand_range_(-0.05f, 0.05f, &env->rng)
    };
    env->body_inertia_scale[0] = rand_range_(0.95f, 1.05f, &env->rng);
    env->body_inertia_scale[1] = rand_range_(0.95f, 1.05f, &env->rng);
    env->body_inertia_scale[2] = rand_range_(0.95f, 1.05f, &env->rng);
    env->ori_delay = rand_log_uniform_(0.5f, 0.95f, &env->rng);
    env->action_delay = rand_log_uniform_(0.1f, 0.5f, &env->rng);

    env->total_mass_local = r->body_mass + env->mass_offset;
    for (int i = 0; i < r->n_dof; i++) {
        env->kp_scale[i]      = rand_range_(0.9f, 1.1f, &env->rng);
        env->kd_scale[i]      = rand_range_(0.9f, 1.1f, &env->rng);
        env->dof_strength[i]  = rand_range_(0.98f, 1.02f, &env->rng);
        env->leg_mass_scale[i]= rand_range_(0.95f, 1.05f, &env->rng);
        env->total_mass_local += r->leg_mass[i] * env->leg_mass_scale[i];
    }

    env->pos = (Vec3){
        rand_range_(-1.0f, 1.0f, &env->rng),
        rand_range_(-1.0f, 1.0f, &env->rng),
        r->sphere_radius + 0.1f
    };
    generate_heightfield(env);
    float roll  = rand_range_(-0.25f, 0.25f, &env->rng);
    float pitch = rand_range_(-0.25f, 0.25f, &env->rng);
    float yaw   = rand_range_(-(float)M_PI, (float)M_PI, &env->rng);
    float cr = cosf(roll*0.5f),  sr = sinf(roll*0.5f);
    float cp = cosf(pitch*0.5f), sp = sinf(pitch*0.5f);
    float cy = cosf(yaw*0.5f),   sy = sinf(yaw*0.5f);
    env->quat = (Quat){
        cr*cp*cy + sr*sp*sy,
        sr*cp*cy - cr*sp*sy,
        cr*sp*cy + sr*cp*sy,
        cr*cp*sy - sr*sp*cy
    };
    env->quat_filt = env->quat;
    env->vel   = (Vec3){0,0,0};
    env->omega = (Vec3){0,0,0};

    for (int i = 0; i < r->n_dof; i++) {
        env->q[i]       = clampf(rand_range_(-0.1f, 0.1f, &env->rng),
                                  r->joint_lo[i], r->joint_hi[i]);
        env->q_visual[i]= env->q[i];
        env->dq[i]      = rand_range_(-0.1f, 0.1f, &env->rng);
        env->dq_comp[i] = 0.0f;
        env->int_err[i] = 0.0f;
        env->last_act[i]= 0.0f;
        env->action_to_use[i] = 0.0f;
    }

    env->cmd_vx = rand_range_(-CMD_VEL_MAX, CMD_VEL_MAX, &env->rng);
    env->cmd_vy = rand_range_(-CMD_VEL_MAX, CMD_VEL_MAX, &env->rng);

    env->tick           = 0;
    env->ep_return      = 0.0f;
    env->ep_vel_err_sum = 0.0f;
    env->stack_head     = 0;
    env->push_timer     = (int)(10.0f / DT_ACT);

    env->force_noise_vel   = (Vec3){0,0,0};
    env->force_noise_omega = (Vec3){0,0,0};

    env->ep_rew_lin_vel = 0.0f;
    env->ep_rew_dof_force = 0.0f;
    env->ep_rew_dof_vel = 0.0f;
    env->ep_rew_dof_limit = 0.0f;
    env->ep_rew_impact = 0.0f;
    env->ep_rew_slip = 0.0f;
    env->ep_rew_action = 0.0f;
    env->ep_rew_action_rate = 0.0f;

    float frame[3 + 2 + MAX_DOF*3 + 9];
    build_frame(env, frame);
    int frame_sz = g_obs_single_frame;
    for (int s = 0; s < N_STACK; s++)
        for (int k = 0; k < frame_sz; k++)
            env->obs_stack[s][k] = frame[k];

    flush_obs(env);
}

void physics_substep(Argus *env, float dt, int substep_idx) {
    const ArgusRobot *r = &g_argus_robot;

    Vec3 F_body   = {0.0f, 0.0f, -env->total_mass_local * GRAVITY};
    Vec3 tau_body = {0.0f, 0.0f, 0.0f};
    float Ixx = r->body_ixx * env->body_inertia_scale[0];
    float Iyy = r->body_iyy * env->body_inertia_scale[1];
    float Izz = r->body_izz * env->body_inertia_scale[2];

    Vec3 com_w = qrot(env->quat, env->com_offset);
    Vec3 grav_vec = {0.0f, 0.0f, -env->total_mass_local * GRAVITY};
    tau_body = v3cross(com_w, grav_vec);

    for (int i = 0; i < r->n_dof; i++) {
        Vec3 dir_body = {r->leg_dir[i][0], r->leg_dir[i][1], r->leg_dir[i][2]};
        Vec3 dir_w    = qrot(env->quat, dir_body);

        float eff_leg_mass = r->leg_mass[i] * env->leg_mass_scale[i];
        float r_eff = r->leg_base_dist[i] + env->q[i];
        float r2    = r_eff * r_eff;
        Ixx += eff_leg_mass * (dir_w.y*dir_w.y + dir_w.z*dir_w.z) * r2;
        Iyy += eff_leg_mass * (dir_w.x*dir_w.x + dir_w.z*dir_w.z) * r2;
        Izz += eff_leg_mass * (dir_w.x*dir_w.x + dir_w.y*dir_w.y) * r2;

        float target = clampf(ACTION_SCALE * env->action_to_use[i],
                              r->joint_lo[i], r->joint_hi[i]);
        float err    = target - env->q[i];
        env->int_err[i] += err * dt;
        float F_raw = KI * env->int_err[i]
                    + (KP * env->kp_scale[i]) * err
                    - (KD * env->kd_scale[i]) * env->dq[i];

        if (substep_idx == 0)
            env->force_target[i] = F_raw;

        float abs_dq = fabsf_(env->dq[i]);
        float F_max  = clampf((V_MOTOR - V_BACK_EMF - abs_dq / R_WHEEL),
                              0.0f, T_MAX) / R_WHEEL;
        float F_app;
        if (env->dq[i] >= 0.0f)
            F_app = (F_raw < F_max) ? F_raw : F_max;
        else
            F_app = (F_raw > -F_max) ? F_raw : -F_max;
        F_app = clampf(F_app, -r->joint_effort[i], r->joint_effort[i]);
        F_app *= env->dof_strength[i];

        env->dq[i] += (F_app - JOINT_DAMP * env->dq[i]) / eff_leg_mass * dt;
        env->q[i]  += env->dq[i] * dt;

        if (env->q[i] < r->joint_lo[i]) {
            env->q[i] = r->joint_lo[i];
            env->dq[i] = fmaxf_(0.0f, env->dq[i]);
        }
        if (env->q[i] > r->joint_hi[i]) {
            env->q[i] = r->joint_hi[i];
            env->dq[i] = (env->dq[i] < 0.0f ? env->dq[i] : 0.0f);
        }

        float fr = r->foot_radius[i];
        float r_foot  = r->leg_base_dist[i] + env->q[i];
        Vec3 foot_pos = v3add(env->pos, v3scale(dir_w, r_foot));

        Vec3 lev_arm    = v3scale(dir_w, r_foot);
        Vec3 v_foot_rot = v3cross(env->omega, lev_arm);
        Vec3 v_foot     = v3add(v3add(env->vel, v_foot_rot),
                                v3scale(dir_w, env->dq[i]));

        float ground_z = sample_heightfield(env, foot_pos.x, foot_pos.y);
        float surface  = ground_z + fr;

        if (foot_pos.z < surface) {
            float pen      = fminf_(surface - foot_pos.z, MAX_PEN);
            float pen_rate = -v_foot.z;
            float N = K_CONTACT * pen + B_CONTACT * fmaxf_(0.0f, pen_rate);
            if (N < 0.0f) N = 0.0f;

            Vec3 v_horiz = {v_foot.x, v_foot.y, 0.0f};
            float spd    = v3norm(v_horiz);
            Vec3 F_fric  = {0,0,0};
            if (spd > 1e-4f) {
                float fric = env->mu_friction * N;
                F_fric = v3scale(v_horiz, -fric / spd);
            }

            Vec3 F_contact = {F_fric.x, F_fric.y, N};
            F_body   = v3add(F_body, F_contact);
            tau_body = v3add(tau_body, v3cross(lev_arm, F_contact));

            float contact_r = sqrtf(fr * pen);
            float tau_coeff = -env->mu_friction * N * contact_r;
            tau_body.x += tau_coeff * env->omega.x;
            tau_body.y += tau_coeff * env->omega.y;
            tau_body.z += tau_coeff * env->omega.z;

            if (N > env->foot_contact_fz[i])
                env->foot_contact_fz[i] = N;
            env->foot_horiz_vel_sq[i] += v_horiz.x*v_horiz.x + v_horiz.y*v_horiz.y;
            env->foot_contact_count[i]++;
        }
    }

    float body_ground = sample_heightfield(env, env->pos.x, env->pos.y);
    if (env->pos.z < body_ground + r->sphere_radius) {
        float pen      = fminf_(body_ground + r->sphere_radius - env->pos.z, MAX_PEN);
        float pen_rate = -env->vel.z;
        float Nz = K_CONTACT * pen + B_CONTACT * fmaxf_(0.0f, pen_rate);
        F_body.z += fmaxf_(0.0f, Nz);
    }

    env->vel = v3add(env->vel, v3scale(F_body, dt / env->total_mass_local));
    env->pos = v3add(env->pos, v3scale(env->vel, dt));

    float gx = (Izz - Iyy) * env->omega.y * env->omega.z;
    float gy = (Ixx - Izz) * env->omega.z * env->omega.x;
    float gz = (Iyy - Ixx) * env->omega.x * env->omega.y;
    env->omega.x += (tau_body.x - gx) / Ixx * dt;
    env->omega.y += (tau_body.y - gy) / Iyy * dt;
    env->omega.z += (tau_body.z - gz) / Izz * dt;
    qintegrate(&env->quat, env->omega, dt);
}

void c_step(Argus *env) {
    const ArgusRobot *r = &g_argus_robot;

    env->terminals[0] = 0.0f;
    env->rewards[0]   = 0.0f;
    env->tick++;

    if (randf_(&env->rng) < CMD_RESAMPLE) {
        env->cmd_vx = rand_range_(-CMD_VEL_MAX, CMD_VEL_MAX, &env->rng);
        env->cmd_vy = rand_range_(-CMD_VEL_MAX, CMD_VEL_MAX, &env->rng);
    }

    for (int i = 0; i < r->n_dof; i++) {
        env->action_to_use[i] = env->actions[i] * (1.0f - env->action_delay)
                               + env->action_delay * env->action_to_use[i];
    }

    memset(env->foot_contact_fz,     0, sizeof(float) * MAX_DOF);
    memset(env->foot_horiz_vel_sq,   0, sizeof(float) * MAX_DOF);
    memset(env->foot_contact_count,  0, sizeof(int)   * MAX_DOF);

    env->push_timer--;
    if (env->push_timer <= 0) {
        env->vel.x   += rand_range_(-1.0f, 1.0f, &env->rng);
        env->vel.y   += rand_range_(-4.0f, 0.0f, &env->rng);
        env->vel.z   += rand_range_(-1.0f, 1.0f, &env->rng);
        env->omega.x += rand_range_(-1.0f, 1.0f, &env->rng);
        env->omega.y += rand_range_(-1.0f, 1.0f, &env->rng);
        env->omega.z += rand_range_(-1.0f, 1.0f, &env->rng);
        env->push_timer = (int)(10.0f / DT_ACT);
    }

    float force_decay = expf(-DT_ACT / 0.2f);
    env->force_noise_vel   = v3scale(env->force_noise_vel, force_decay);
    env->force_noise_omega = v3scale(env->force_noise_omega, force_decay);
    float force_prob = rand_log_uniform_(0.001f, 0.1f, &env->rng);
    if (randf_(&env->rng) < force_prob) {
        float scale = 0.05f * env->total_mass_local * GRAVITY;
        env->force_noise_vel = (Vec3){
            rand_range_(-1.0f, 1.0f, &env->rng) * scale,
            rand_range_(-1.0f, 1.0f, &env->rng) * scale,
            rand_range_(-1.0f, 1.0f, &env->rng) * scale
        };
        env->force_noise_omega = (Vec3){
            rand_range_(-1.0f, 1.0f, &env->rng) * scale * 0.1f,
            rand_range_(-1.0f, 1.0f, &env->rng) * scale * 0.1f,
            rand_range_(-1.0f, 1.0f, &env->rng) * scale * 0.1f
        };
    }
    float inv_mass = 1.0f / env->total_mass_local;
    env->vel   = v3add(env->vel, v3scale(env->force_noise_vel, DT_ACT * inv_mass));
    env->omega = v3add(env->omega, v3scale(env->force_noise_omega, DT_ACT));

    float q_pre[MAX_DOF];
    for (int i = 0; i < r->n_dof; i++) q_pre[i] = env->q[i];

    for (int sub = 0; sub < DECIMATION; sub++)
        physics_substep(env, DT_SIM, sub);

    float dt_act_inv = 1.0f / DT_ACT;
    for (int i = 0; i < r->n_dof; i++) {
        float dq_raw = (env->q[i] - q_pre[i]) * dt_act_inv;
        env->dq_comp[i] = DQ_ALPHA * dq_raw + (1.0f - DQ_ALPHA) * env->dq_comp[i];
        env->q_visual[i] += (env->q[i] - env->q_visual[i]) * 0.3f;
    }

    float ex = env->vel.x - env->cmd_vx;
    float ey = env->vel.y - env->cmd_vy;
    float ez = env->vel.z;
    float rew_lin_vel = expf(-5.0f * (ex*ex + ey*ey) - 0.5f * (ez*ez));

    float force_penalty_sum = 0.0f;
    float soft_lo = -0.3f * F_MAX_ABS;
    float soft_hi =  0.3f * F_MAX_ABS;
    for (int i = 0; i < r->n_dof; i++) {
        float ft = env->force_target[i];
        float oob = ft - clampf(ft, soft_lo, soft_hi);
        float norm = oob / F_MAX_ABS;
        force_penalty_sum += norm * norm;
    }
    float rew_dof_force = expf(force_penalty_sum * -0.1f);

    float vel_sum = 0.0f;
    for (int i = 0; i < r->n_dof; i++)
        vel_sum += env->dq_comp[i] * env->dq_comp[i];
    float rew_dof_vel = expf(vel_sum * -0.2f);

    float limit_sum = 0.0f;
    for (int i = 0; i < r->n_dof; i++) {
        float lo = r->joint_lo[i] + 0.05f;
        float hi = r->joint_hi[i] - 0.05f;
        float v = env->q[i] - clampf(env->q[i], lo, hi);
        limit_sum += v * v;
    }
    float rew_dof_limit = limit_sum;

    float gravity_inv = 1.0f / (env->total_mass_local * GRAVITY);
    float impact_sum = 0.0f;
    for (int i = 0; i < r->n_dof; i++) {
        float nf = clampf(env->foot_contact_fz[i] * gravity_inv - 1.0f,
                          0.0f, 2.0f);
        impact_sum += nf * nf;
    }
    float rew_impact = impact_sum;

    float slip_sum = 0.0f;
    for (int i = 0; i < r->n_dof; i++) {
        if (env->foot_contact_count[i] > 0)
            slip_sum += env->foot_horiz_vel_sq[i]
                      / (float)env->foot_contact_count[i];
    }
    float rew_slip = slip_sum;

    float act_oob_sum = 0.0f;
    for (int i = 0; i < r->n_dof; i++) {
        float oob = env->actions[i] - clampf(env->actions[i], -0.5f, 0.5f);
        act_oob_sum += oob * oob;
    }
    float rew_action = expf(act_oob_sum * -0.2f);

    float rate_sum = 0.0f;
    for (int i = 0; i < r->n_dof; i++) {
        float rate = (env->actions[i] - env->last_act[i]) * dt_act_inv;
        rate_sum += rate * rate;
    }
    float rew_action_rate = expf(rate_sum * -2e-4f);

    float reward = 1.0f  * rew_lin_vel
                 + 0.4f  * rew_dof_force
                 + 0.1f  * rew_dof_vel
                 - 1.0f  * rew_dof_limit
                 - 0.01f * rew_impact
                 - 0.01f * rew_slip
                 + 0.2f  * rew_action
                 + 0.2f  * rew_action_rate;
    if (reward < 0.0f) reward = 0.0f;

    env->rewards[0]      = reward;
    env->ep_return      += reward;
    env->ep_vel_err_sum += sqrtf(ex*ex + ey*ey);

    env->ep_rew_lin_vel     += rew_lin_vel;
    env->ep_rew_dof_force   += rew_dof_force;
    env->ep_rew_dof_vel     += rew_dof_vel;
    env->ep_rew_dof_limit   += rew_dof_limit;
    env->ep_rew_impact      += rew_impact;
    env->ep_rew_slip        += rew_slip;
    env->ep_rew_action      += rew_action;
    env->ep_rew_action_rate += rew_action_rate;

    for (int i = 0; i < r->n_dof; i++) env->last_act[i] = env->actions[i];

    env->quat_filt = qslerp(env->ori_delay, env->quat, env->quat_filt);

    build_frame(env, env->obs_stack[env->stack_head]);
    env->stack_head = (env->stack_head + 1) % N_STACK;
    flush_obs(env);

    float crash_ground = sample_heightfield(env, env->pos.x, env->pos.y);
    int base_crashed = (env->pos.z < crash_ground + r->sphere_radius * 0.5f);
    int timeout      = (env->tick >= MAX_STEPS);
    if (base_crashed || timeout) {
        env->terminals[0] = 1.0f;
        float mean_vel_err = env->ep_vel_err_sum / (float)env->tick;
        float sc = clampf(1.0f - mean_vel_err / CMD_VEL_MAX, 0.0f, 1.0f);
        env->log.episode_return  += env->ep_return;
        env->log.episode_length  += (float)env->tick;
        env->log.mean_vel_error  += mean_vel_err;
        env->log.score           += sc;
        env->log.perf            += sc;
        env->log.rew_lin_vel     += env->ep_rew_lin_vel;
        env->log.rew_dof_force   += env->ep_rew_dof_force;
        env->log.rew_dof_vel     += env->ep_rew_dof_vel;
        env->log.rew_dof_limit   += env->ep_rew_dof_limit;
        env->log.rew_impact      += env->ep_rew_impact;
        env->log.rew_slip        += env->ep_rew_slip;
        env->log.rew_action      += env->ep_rew_action;
        env->log.rew_action_rate += env->ep_rew_action_rate;
        env->log.n++;
        c_reset(env);
    }
}

#define WIN_W 1200
#define WIN_H 800

static Camera3D g_camera;
static float g_cam_distance = 3.0f;
static float g_cam_azimuth  = -0.8f;   // radians
static float g_cam_elevation = 0.35f;  // radians
static int   g_cam_dragging = 0;
static Vector2 g_cam_last_mouse = {0};

static void update_cam(Vec3 target) {
    float r = g_cam_distance;
    float az = g_cam_azimuth;
    float el = g_cam_elevation;
    g_camera.position = (Vector3){
        target.x + r * cosf(el) * cosf(az),
        target.y + r * cosf(el) * sinf(az),
        target.z + r * sinf(el)
    };
    g_camera.target = (Vector3){target.x, target.y, target.z};
}

static void handle_cam_input(Vec3 target) {
    Vector2 mouse = GetMousePosition();

    if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
        g_cam_dragging = 1;
        g_cam_last_mouse = mouse;
    }
    if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT))
        g_cam_dragging = 0;

    if (g_cam_dragging && IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
        float dx = mouse.x - g_cam_last_mouse.x;
        float dy = mouse.y - g_cam_last_mouse.y;
        g_cam_azimuth   -= dx * 0.005f;
        g_cam_elevation += dy * 0.005f;
        g_cam_elevation  = clampf(g_cam_elevation, 0.05f, 1.5f);
        g_cam_last_mouse = mouse;
    }

    float wheel = GetMouseWheelMove();
    if (wheel != 0.0f) {
        g_cam_distance -= wheel * 0.3f;
        g_cam_distance  = clampf(g_cam_distance, 0.5f, 20.0f);
    }

    update_cam(target);
}

void c_render(Argus *env) {
    if (!IsWindowReady()) {
        SetConfigFlags(FLAG_MSAA_4X_HINT);
        InitWindow(WIN_W, WIN_H, "PufferLib Argus (3D)");
        SetTargetFPS(30);

        g_camera.up = (Vector3){0.0f, 0.0f, 1.0f};
        g_camera.fovy = 45.0f;
        g_camera.projection = CAMERA_PERSPECTIVE;
        update_cam(env->pos);
    }

    if (IsKeyDown(KEY_ESCAPE)) exit(0);

    const ArgusRobot *r = &g_argus_robot;

    handle_cam_input(env->pos);

    BeginDrawing();
    ClearBackground((Color){15, 20, 30, 255});
    BeginMode3D(g_camera);

    int hf_step = 2;
    for (int i = 0; i < HF_RES - hf_step; i += hf_step) {
        for (int j = 0; j < HF_RES - hf_step; j += hf_step) {
            float h = env->heightfield[i][j];
            if (fabsf_(h) < 0.005f) continue;
            float wx = env->hf_origin_x + (i - HF_RES * 0.5f) * HF_SCALE;
            float wy = env->hf_origin_y + (j - HF_RES * 0.5f) * HF_SCALE;
            float s = HF_SCALE * hf_step;
            Color tc;
            float mh = fmaxf_(env->terrain_height, 0.01f);
            if (h > 0.0f)
                tc = (Color){60 + (int)(h/mh*80), 40, 30, 255};
            else
                tc = (Color){20, 30, 50 + (int)(-h/mh*60), 255};
            DrawCube((Vector3){wx + s*0.5f, wy + s*0.5f, h*0.5f},
                     s, s, fabsf_(h) + 0.002f, tc);
        }
    }
    {
        float spacing = 1.0f;
        float ext = 20.0f;
        float cx = floorf(env->pos.x / spacing) * spacing;
        float cy = floorf(env->pos.y / spacing) * spacing;
        Color grid_color = {35, 42, 52, 255};
        for (float g = -ext; g <= ext; g += spacing) {
            DrawLine3D((Vector3){cx + g, cy - ext, 0.0f},
                       (Vector3){cx + g, cy + ext, 0.0f}, grid_color);
            DrawLine3D((Vector3){cx - ext, cy + g, 0.0f},
                       (Vector3){cx + ext, cy + g, 0.0f}, grid_color);
        }
    }

    Vector3 body_pos = {env->pos.x, env->pos.y, env->pos.z};

    DrawSphere(body_pos, r->sphere_radius, (Color){0, 160, 200, 220});
    DrawSphereWires(body_pos, r->sphere_radius, 8, 12, (Color){0, 200, 255, 100});

    {
        Vec3 fwd_body = {1.0f, 0.0f, 0.0f};
        Vec3 fwd_w = qrot(env->quat, fwd_body);
        Vector3 hdg_end = {
            body_pos.x + fwd_w.x * r->sphere_radius * 1.5f,
            body_pos.y + fwd_w.y * r->sphere_radius * 1.5f,
            body_pos.z + fwd_w.z * r->sphere_radius * 1.5f
        };
        DrawCylinderEx(body_pos, hdg_end, 0.012f, 0.003f, 6, (Color){255, 220, 0, 255});
    }

    for (int i = 0; i < r->n_dof; i++) {
        Vec3 dir_body = {r->leg_dir[i][0], r->leg_dir[i][1], r->leg_dir[i][2]};
        Vec3 dir_w    = qrot(env->quat, dir_body);

        float r_foot = r->leg_base_dist[i] + env->q_visual[i];
        Vector3 foot_tip = {
            body_pos.x + dir_w.x * r_foot,
            body_pos.y + dir_w.y * r_foot,
            body_pos.z + dir_w.z * r_foot
        };

        float fr = r->foot_radius[i];
        float foot_ground = sample_heightfield(env, foot_tip.x, foot_tip.y);
        int in_contact = (foot_tip.z < foot_ground + fr + 0.02f);

        Color leg_color = in_contact ? (Color){0, 220, 120, 255} : (Color){100, 130, 170, 200};
        DrawCylinderEx(body_pos, foot_tip, 0.015f, 0.008f, 8, leg_color);

        Color foot_color = in_contact ? (Color){0, 255, 140, 255} : (Color){150, 170, 200, 200};
        DrawSphere(foot_tip, fr, foot_color);
    }

    {
        float arrow_scale = 0.5f;
        Vector3 vel_end = {
            body_pos.x + env->vel.x * arrow_scale,
            body_pos.y + env->vel.y * arrow_scale,
            body_pos.z + env->vel.z * arrow_scale
        };
        DrawCylinderEx(body_pos, vel_end, 0.015f, 0.005f, 6, (Color){0, 230, 60, 255});
    }

    {
        float arrow_scale = 0.5f;
        Vector3 cmd_end = {
            body_pos.x + env->cmd_vx * arrow_scale,
            body_pos.y + env->cmd_vy * arrow_scale,
            body_pos.z
        };
        DrawCylinderEx(body_pos, cmd_end, 0.012f, 0.004f, 6, (Color){230, 80, 80, 255});
    }

    EndMode3D();

    int y = 10;
    DrawText(TextFormat("step  %d/%d", env->tick, MAX_STEPS), 10, y, 18, RAYWHITE);
    y += 22;
    DrawText(TextFormat("vel   %.2f  %.2f  %.2f", env->vel.x, env->vel.y, env->vel.z),
             10, y, 18, GREEN);
    y += 22;
    DrawText(TextFormat("cmd   %.2f  %.2f", env->cmd_vx, env->cmd_vy), 10, y, 18, RED);
    y += 22;
    DrawText(TextFormat("pos.z %.3f m", env->pos.z), 10, y, 18, YELLOW);
    y += 22;
    DrawText(TextFormat("ret   %.1f", env->ep_return), 10, y, 18, RAYWHITE);
    y += 22;
    DrawText(TextFormat("DOF   %d  mu=%.2f  dm=%.1f", r->n_dof,
             env->mu_friction, env->mass_offset), 10, y, 18, (Color){150,150,150,255});
    y += 22;

    float t = fmaxf_(1.0f, (float)env->tick);
    DrawText(TextFormat("lv %.3f  df %.3f  dv %.3f  dl %.4f",
             env->ep_rew_lin_vel/t, env->ep_rew_dof_force/t,
             env->ep_rew_dof_vel/t, env->ep_rew_dof_limit/t),
             10, y, 14, (Color){180,180,100,255});
    y += 18;
    DrawText(TextFormat("im %.4f  sl %.4f  ac %.3f  ar %.3f",
             env->ep_rew_impact/t, env->ep_rew_slip/t,
             env->ep_rew_action/t, env->ep_rew_action_rate/t),
             10, y, 14, (Color){180,180,100,255});
    y += 22;

    DrawText("Leg extensions:", 10, y, 14, (Color){120,120,120,255});
    y += 18;
    int bar_w = 60, bar_h = 8;
    for (int i = 0; i < r->n_dof && i < 20; i++) {
        int col = i / 10;
        int row = i % 10;
        int bx = 10 + col * (bar_w + 50);
        int by = y + row * (bar_h + 3);

        float pct = (env->q[i] - r->joint_lo[i]) / (r->joint_hi[i] - r->joint_lo[i]);
        pct = clampf(pct, 0.0f, 1.0f);
        int fill = (int)(pct * bar_w);

        DrawText(TextFormat("%2d", i), bx, by, 12, (Color){100,100,100,255});
        DrawRectangle(bx + 22, by, bar_w, bar_h, (Color){30,30,30,255});
        Color bar_c = (pct > 0.8f) ? (Color){255,100,60,255} :
                      (pct > 0.4f) ? (Color){100,200,100,255} : (Color){60,120,200,255};
        DrawRectangle(bx + 22, by, fill, bar_h, bar_c);
    }

    y += 10 * (bar_h + 3) + 10;
    DrawText("Mouse: orbit | Scroll: zoom | ESC: quit", 10, y, 14, (Color){80,80,80,255});

    EndDrawing();
}

void c_close(Argus *env) {
    (void)env;
    if (IsWindowReady()) CloseWindow();
}
