#include <stdio.h>

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <assert.h>

#include "clebsch_gordan.h"


#include "e3nn.h"

#include <math.h>
#include <complex.h>
#include <stdlib.h>
#include <stdio.h>


#define EPS 1e-8


#include "tp.c"

// cache for common factorials - makes initial cg computation a little faster
#define MAX_FACT_CACHE 9
double fact_cache[10] = {
  1.00000000000000000000e+00,
  1.00000000000000000000e+00,
  2.00000000000000000000e+00,
  6.00000000000000000000e+00,
  2.40000000000000000000e+01,
  1.20000000000000000000e+02,
  7.20000000000000000000e+02,
  5.04000000000000000000e+03,
  4.03200000000000000000e+04,
  3.62880000000000000000e+05,
};

#define MAX_DFACT_CACHE 9
double dfact_cache[10] = {
  1.00000000000000000000e+00,
  1.00000000000000000000e+00,
  2.00000000000000000000e+00,
  3.00000000000000000000e+00,
  8.00000000000000000000e+00,
  1.50000000000000000000e+01,
  4.80000000000000000000e+01,
  1.05000000000000000000e+02,
  3.84000000000000000000e+02,
  9.45000000000000000000e+02,
};


typedef float***** ClebschGordanCache;
typedef SparseClebschGordanMatrix** SparseClebschGordanCache;

// TODO: nicer way to avoid globals?
SparseClebschGordanCache* sparse_cg_cache = NULL;
ClebschGordanCache* cg_cache = NULL;

double factorial(int n) {
    if(n < MAX_FACT_CACHE) { return fact_cache[n]; }
    double x = (double) n;
    while(--n > MAX_FACT_CACHE) {
        x *= (double) n;
    }
    return x * fact_cache[n];
}

double dfactorial(int n) {
    if(n < MAX_DFACT_CACHE) { return dfact_cache[n]; }
    double x = (double) n;
    while((n -= 2) > MAX_FACT_CACHE) {
        x *= (double) n;
    }
    return x * dfact_cache[n];
}


double _su2_cg(int j1, int j2, int j3, int m1, int m2, int m3) {
    // calculate the Clebsch-Gordon coefficient
    // for SU(2) coupling (j1,m1) and (j2,m2) to give (j3,m3).
    // based on e3nn.o3._wigner._su2_clebsch_gordan_coeff
    if (m3 != m1 + m2) {
        return 0;
    }
    int vmin = MAX(MAX(-j1 + j2 + m3, -j1 + m1), 0);
    int vmax = MIN(MIN(j2 + j3 + m1, j3 - j1 + j2), j3 + m3);
    
    double C = sqrt(
        (double) (2 * j3 + 1) *
        (
            (factorial(j3 + j1 - j2) *
                factorial(j3 - j1 + j2) *
                factorial(j1 + j2 - j3) *
                factorial(j3 + m3) *
                factorial(j3 - m3)
            ) /
            (factorial(j1 + j2 + j3 + 1) *
                factorial(j1 - m1) *
                factorial(j1 + m1) *
                factorial(j2 - m2) *
                factorial(j2 + m2)
            )
        )
    );
    double S = 0;
    for (int v = vmin; v <= vmax; v++) {
        S += pow(-1, v + j2 + m2) * (
            (factorial(j2 + j3 + m1 - v) * factorial(j1 - m1 + v)) /
            (factorial(v) * factorial(j3 - j1 + j2 - v) * factorial(j3 + m3 - v) * factorial(v + j1 - j2 - m3))
        );
    }
    C = C * S;
    return C;
}


float complex change_basis_real_to_complex(int l, int m1, int m2) {
    // https://en.wikipedia.org/wiki/Spherical_harmonics#Real_form
    // based on:
    // https://github.com/e3nn/e3nn-jax/blob/a2a81ab451b9cd597d7be27b3e1faba79457475d/e3nn_jax/_src/so3.py#L6
    // but instead of returning a matrix, just index at [m1, m2], e.g.:
    // change_basis_real_to_complex(l, m1, m2) = e3nn_jax._src.so3.change_basis_real_to_complex(l)[m1, m2]
    const float complex factor = cpow(-I, l);
    const float inv_sqrt2 = 1 / sqrt(2);
    const float complex I_inv_sqrt2 = I * inv_sqrt2;
    
    if (m1 == m2 && m2 == 0) { return factor; }
    if (m1 < 0) {
        if (m2 == -m1) { return inv_sqrt2 * factor; }
        if (m2 == m1) { return -I_inv_sqrt2 * factor; }
    }
    if (m1 > 0) {
        if (m2 == m1) { return pow(-1, m1) * inv_sqrt2 * factor; }
        if (m2 == -m1) { return pow(-1, m1) * I_inv_sqrt2 * factor; }
    }
    return 0;
}


float compute_clebsch_gordan(int l1, int l2, int l3, int m1, int m2, int m3) {
    // Clebsch-Gordan coefficients of the real irreducible representations of
    // SO3, based on:
    // https://github.com/e3nn/e3nn-jax/blob/a2a81ab451b9cd597d7be27b3e1faba79457475d/e3nn_jax/_src/so3.py#L21
    float c = 0;
    for (int i = -l1; i <= l1; i++) {
        for (int j = -l2; j <= l2; j++) {
            for (int k = -l3; k <= l3; k++) {
                c += creal(
                    change_basis_real_to_complex(l1, i, m1)
                    * change_basis_real_to_complex(l2, j, m2)
                    * conj(change_basis_real_to_complex(l3, k, m3))
                    * _su2_cg(l1, l2, l3, i, j, k)
                );
            }
        }
    }
    // note that this normalization is applied in the su2_clebsch_gordan in the
    // JAX library, however we are not using that function and call _su2_cg directly
    return c / sqrt(2 * l3 + 1);
}


float clebsch_gordan(int l1, int l2, int l3, int m1, int m2, int m3) {
    // Clebsch-Gordan coefficients of the real irreducible representations of SO3
    return cg_cache[l1][l2][l3][m1 + l1][m2 + l2][m3 + l3];
}

SparseClebschGordanMatrix sparse_clebsch_gordan(int l1, int l2, int l3) {
    return sparse_cg_cache[l1][l2][l3];
}


void build_clebsch_gordan_cache(void) {
    // precompute all Clebsch-Gordan coefficients up to L_MAX
    // NOTE: only computing l1 and l2 up to L_MAX / 2 for now to make things faster 
    if (cg_cache) {
        // already built
        return;
    }
    cg_cache = (ClebschGordanCache*) malloc((L_MAX / 2 + 1) * sizeof(float*****));
    for (int l1 = 0; l1 <= L_MAX / 2; l1++) { 
        cg_cache[l1] = (float*****) malloc((L_MAX / 2 + 1) * sizeof(float****));
        for (int l2 = 0; l2 <= L_MAX / 2; l2++) {
            cg_cache[l1][l2] = (float****) malloc((L_MAX + 1) * sizeof(float***));
            for (int l3 = abs(l1 - l2); l3 <= l1 + l2 && l3 <= L_MAX; l3++) {
                cg_cache[l1][l2][l3] = (float***) malloc((2 * l1 + 1) * sizeof(float**));
                for (int m1 = -l1; m1 <= l1; m1++) {
                    cg_cache[l1][l2][l3][m1 + l1] = (float**) malloc((2 * l2 + 1) * sizeof(float*));
                    for (int m2 = -l2; m2 <= l2; m2++) {
                        cg_cache[l1][l2][l3][m1 + l1][m2 + l2] = (float*) malloc((2 * l3 + 1) * sizeof(float));
                        for (int m3 = -l3; m3 <= l3; m3++) {
                            cg_cache[l1][l2][l3][m1 + l1][m2 + l2][m3 + l3] = compute_clebsch_gordan(l1, l2, l3, m1, m2, m3);
                        }
                    }
                }
            }
        }
    }
}

static void free_clebsch_gordan_cache(void) { 
    if (!cg_cache) {
        return;
    }
    for (int l1 = 0; l1 <= L_MAX / 2; l1++) {
        for (int l2 = 0; l2 <= L_MAX / 2; l2++) {
            for (int l3 = abs(l1 - l2); l3 <= l1 + l2 && l3 <= L_MAX; l3++) {
                for (int m1 = -l1; m1 <= l1; m1++) {
                    for (int m2 = -l2; m2 <= l2; m2++) {
                        free(cg_cache[l1][l2][l3][m1 + l1][m2 + l2]);
                    }
                    free(cg_cache[l1][l2][l3][m1 + l1]);
                }
                free(cg_cache[l1][l2][l3]);
            }
            free(cg_cache[l1][l2]);
        }
        free(cg_cache[l1]);
    }
    free(cg_cache);
}

void build_sparse_clebsch_gordan_cache(void) {
    // build sparse Clebsch-Gordan cache
    // NOTE: only computing l1 and l2 up to L_MAX / 2 for now to make things faster 
    if (sparse_cg_cache) {
        // already built
        return;
    }
    build_clebsch_gordan_cache();
    sparse_cg_cache = (SparseClebschGordanMatrix***) malloc((L_MAX / 2 + 1) * sizeof(SparseClebschGordanMatrix**));
    for (int l1 = 0; l1 <= L_MAX / 2; l1++) { 
        sparse_cg_cache[l1] = (SparseClebschGordanMatrix**) malloc((L_MAX / 2 + 1) * sizeof(SparseClebschGordanMatrix*));
        for (int l2 = 0; l2 <= L_MAX / 2; l2++) {
            sparse_cg_cache[l1][l2] = (SparseClebschGordanMatrix*) malloc((L_MAX + 1) * sizeof(SparseClebschGordanMatrix));
            for (int l3 = abs(l1 - l2); l3 <= l1 + l2 && l3 <= L_MAX; l3++) {

                int size = 0;
                for (int m1 = -l1; m1 <= l1; m1++) {
                    for (int m2 = -l2; m2 <= l2; m2++) {
                        for (int m3 = -l3; m3 <= l3; m3++) {
                            float c = clebsch_gordan(l1, l2, l3, m1, m2, m3);
                            if (fabs(c) > EPS) {
                                size++;
                            }
                        }
                    }
                }
                sparse_cg_cache[l1][l2][l3].elements = (SparseClebschGordanElement*) malloc(size * sizeof(SparseClebschGordanElement));
                sparse_cg_cache[l1][l2][l3].size = size;

                int index = 0;
                for (int m1 = -l1; m1 <= l1; m1++) {
                    for (int m2 = -l2; m2 <= l2; m2++) {
                        for (int m3 = -l3; m3 <= l3; m3++) {
                            float c = clebsch_gordan(l1, l2, l3, m1, m2, m3);
                            if (fabs(c) > EPS) {
                                sparse_cg_cache[l1][l2][l3].elements[index].m1 = m1;
                                sparse_cg_cache[l1][l2][l3].elements[index].m2 = m2;
                                sparse_cg_cache[l1][l2][l3].elements[index].m3 = m3;
                                sparse_cg_cache[l1][l2][l3].elements[index].c = c;
                                index++;
                            }
                        }
                    }
                }
            }
        }
    }
}

static void free_sparse_clebsch_gordan_cache(void) {
    if (!sparse_cg_cache) {
        return;
    }
    for (int l1 = 0; l1 <= L_MAX / 2; l1++) {
        for (int l2 = 0; l2 <= L_MAX / 2; l2++) {
            for (int l3 = abs(l1 - l2); l3 <= l1 + l2 && l3 <= L_MAX; l3++) {
                free(sparse_cg_cache[l1][l2][l3].elements);
            }
            free(sparse_cg_cache[l1][l2]);
        }
        free(sparse_cg_cache[l1]);
    }
    free(sparse_cg_cache);
}

__attribute__((destructor))
static void cleanup_clebsch_gordan_cache(void) {
    // ensure caches are freed after program completion
    free_clebsch_gordan_cache();
    free_sparse_clebsch_gordan_cache();
}


// index of spherical harmonic (l, m) in array
#define SH_IDX(l, m) ((l) * (l) + (m) + (l))

// void scan_irrep(const char * irrep, int * c, int * l, char * p, int *idx) {
//     // parse irrep string into c, l, p
//     // single irrep lenght = 7 "1x1e + " 
//     int index = *idx;
//     char _c = irrep[index];
//     char _l = irrep[index + 2];
//     printf("c: %c, l: %c, p: %d\n", _c, _l, index);

//     *c = _c - '0';
//     *l = _l - '0';
//     *p = irrep[index + 3];

//     // print processed substring
//     char *sub = (char *) malloc(7 * sizeof(char));
//     strncpy(sub, irrep + index, 4);
//     printf("irrep: %s\n", sub);
//     free(sub);
//     printf("c: %d, l: %d, p: %c\n", *c, *l, *p);
//     if(irrep[index + 4] == '\0') {
//         *idx = -1;
//     } else {
//         *idx = index + 7;
//     }
// }
void scan_irrep(const char * irrep, int * c, int * l, char * p, int *idx) {
    // parse irrep string into c, l, p
    // single irrep lenght = 7 "1x1e + " 
    int index = *idx;
    char _c = irrep[index];
    char _l = irrep[index + 2];
    // printf("c: %c, l: %c, idx: %d\n", _c, _l, index);

    *c = _c - '0';
    *l = _l - '0';
    *p = irrep[index + 3];

    // print processed substring
    // char *sub = (char *) malloc(7 * sizeof(char));
    // strncpy(sub, irrep + index, 4);
    // printf("irrep: %s\n", sub);
    // free(sub);
    // printf("c: %d, l: %d, p: %c\n", *c, *l, *p);
    if(irrep[index + 4] == '\0') {
        *idx = -1;
    } else {
        *idx = index + 7;
    }
}

Irreps* irreps_create(const char* str) {
    // parse str into an array of Irrep with length size
    Irreps* irreps = (Irreps*) malloc(sizeof(Irreps));
    irreps->size = 1;
    for (const char* p = str; *p; p++) {
        if (*p == '+') {
            irreps->size++;
        }
    }
    irreps->irreps = (Irrep*) malloc(irreps->size * sizeof(Irrep));

    int c, l, index = 0, idx = 0;
    char p;
    const char* start = str;
    // printf("start: %s\n", start);
    while (idx != -1) {
        scan_irrep(start, &c, &l, &p, &idx);
        // printf("idx: %d\n", idx);
        irreps->irreps[index++] = (Irrep){ c, l, (p == 'e') ? EVEN : ODD };
        // start = strchr(start, '+');
        // if (!start) break;
        // start++;
    }
    return irreps;
}


Irreps* irreps_copy(const Irreps* irreps) {
    Irreps* copy = (Irreps*) malloc(sizeof(Irreps));
    copy->size = irreps->size;
    copy->irreps = (Irrep*) malloc(copy->size * sizeof(Irrep));
    memcpy(copy->irreps, irreps->irreps, copy->size * sizeof(Irrep));
    return copy;
}


Irreps* irreps_tensor_product(const Irreps* irreps_1, const Irreps* irreps_2) {
    // Lookup table for channel count for each irrep in output
    // indexed by (l + (p+1)/2 * (L_MAX + 1))
    int c_count[(L_MAX + 1) * 2] = {0};
    for (int i1 = 0; i1 < irreps_1->size; i1++) {
        int l1 = irreps_1->irreps[i1].l;
        int c1 = irreps_1->irreps[i1].c;
        int p1 = irreps_1->irreps[i1].p;
        for (int i2 = 0; i2 < irreps_2->size; i2++) {
            int l2 = irreps_2->irreps[i2].l;
            int c2 = irreps_2->irreps[i2].c;
            int p2 = irreps_2->irreps[i2].p;
            int po = p1 * p2;
            int lo_min = abs(l1 - l2);
            int lo_max = l1 + l2;
            for (int lo = lo_min; lo <= lo_max; lo++) {
                c_count[lo + (po + 1) / 2 * (L_MAX + 1)] += c1 * c2;
            }
        }
    }
    Irreps* irreps = (Irreps*) malloc(sizeof(Irreps));
    irreps->size = 0;
    for (int i = 0; i < (L_MAX + 1) * 2; i++) {
        if (c_count[i] != 0) {
            irreps -> size++;
        }
    }
    irreps->irreps = (Irrep*) malloc(irreps->size * sizeof(Irrep));
    int index = 0;
    for (int l = 0; l <= L_MAX; l++) {
        int p = l % 2 == 0 ? EVEN : ODD;
        int c = c_count[l + (p + 1) / 2 * (L_MAX + 1)];
        if (c > 0) {
            irreps->irreps[index++] = (Irrep){ c, l, p };
        }
        p *= -1;
        c = c_count[l + (p + 1) / 2 * (L_MAX + 1)];
        if (c > 0) {
            irreps->irreps[index++] = (Irrep){ c, l, p };
        }
    }
    return irreps;
}


int irrep_compare(const Irrep* i1, const Irrep* i2) {
    if (i1->l == i2->l) {
        if (i1->p == i2->p) {
            return 0;
        } else if ((i1->l % 2 == 0 && i1->p == EVEN) || 
                   (i1->l % 2 == 1 && i1->p == ODD)) {
            return -1;
        } else {
            return 1;
        }
    } else {
        return i1->l - i2->l;
    }
}


bool irreps_is_sorted(const Irreps* irreps) {
    if (irreps->size < 2) { return true; }
    for (int i = 1; i < irreps->size; i++) {
        if (irrep_compare(&irreps->irreps[i-1], &irreps->irreps[i]) >= 0) {
            return false;
        }
    }
    return true;
}


Irreps* irreps_concatenate(const Irreps* irreps_1, const Irreps* irreps_2) {
    assert(irreps_is_sorted(irreps_1));
    assert(irreps_is_sorted(irreps_2));
    Irreps* irreps = (Irreps*) malloc(sizeof(Irreps));
    irreps->size = 0;
    // allocate for worst case and realloc later
    irreps->irreps = (Irrep*) malloc((irreps_1->size + irreps_2->size) * sizeof(Irrep));
    int i1 = 0, i2 = 0;
    while (i1 < irreps_1->size || i2 < irreps_2->size) {
        Irrep write_irr;
        if (
            (i1 < irreps_1->size) &&
            (i2 >= irreps_2->size || irrep_compare(&irreps_1->irreps[i1], &irreps_2->irreps[i2]) <= 0)) {
            write_irr = irreps_1->irreps[i1++];
        } else {
            write_irr = irreps_2->irreps[i2++];
        }

        if (irreps->size == 0 || irrep_compare(&write_irr, &irreps->irreps[irreps->size - 1]) > 0) {
            irreps->irreps[irreps->size] = write_irr;
            irreps->size += 1;
        } else {
            irreps->irreps[irreps->size - 1].c += write_irr.c;
        }
    }
    irreps->irreps = (Irrep*) realloc(irreps->irreps, irreps->size * sizeof(Irrep));
    return irreps;
}


Irreps* irreps_linear(const Irreps* irreps_in, const Irreps* irreps_out, const bool force_irreps_out) {
    Irreps* irreps = (Irreps*) malloc(sizeof(Irreps));
    irreps->size = 0;
    irreps->irreps = (Irrep*) malloc(irreps_out->size * sizeof(Irrep));

    for (int i_out = 0; i_out < irreps_out->size; i_out++) {
        if (force_irreps_out) {
            irreps->irreps[irreps->size++] = irreps_out->irreps[i_out];
            continue;
        }
        for (int i_in = 0; i_in < irreps_in->size; i_in++) {
            if (irrep_compare(&irreps_in->irreps[i_in], &irreps_out->irreps[i_out]) == 0) {
                irreps->irreps[irreps->size++] = irreps_out->irreps[i_out];
                break;
            }
        }
    }
    if (!force_irreps_out) {
        irreps->irreps = (Irrep*) realloc(irreps->irreps, irreps->size * sizeof(Irrep));
    }
    return irreps;
}


int linear_weight_size(const Irreps* irreps_in, const Irreps* irreps_out) {
    int size = 0;
    for (int i_in = 0; i_in < irreps_in->size; i_in++) {
        for (int i_out = 0; i_out < irreps_out->size; i_out++) {
            if (irrep_compare(&irreps_in->irreps[i_in], &irreps_out->irreps[i_out]) == 0) {
                size += irreps_in->irreps[i_in].c * irreps_out->irreps[i_out].c;
            }
        }
    }
    return size;
}


void irreps_free(Irreps* irreps) {
    free(irreps->irreps);
    free(irreps);
}


int irrep_dim(const Irrep* irr) {
    return irr->c * (2 * irr->l + 1);
}


void irreps_print(const Irreps* irreps) {
    for (int i = 0; i < irreps->size; i++) {
        printf("%dx%d%s", irreps->irreps[i].c, irreps->irreps[i].l, irreps->irreps[i].p == EVEN ? "e" : "o");
        if (i < irreps->size - 1) {
            printf(" + ");
        }
    }
    printf("\n");
}

    
int irreps_dim(const Irreps* irreps) {
    int dim = 0;
    for (int i = 0; i < irreps->size; i++) {
        dim += irrep_dim(&irreps->irreps[i]);
    }
    return dim;
}




void tensor_product_v1(const Irreps* irreps_1, const float* data_1, const Irreps* irreps_2, const float* data_2, const Irreps* irreps_o, float* data_o) {
    build_clebsch_gordan_cache();

    // Lookup table where the start of each out irrep will be in datao
    // only need to store one location per l/parity pair, it is cheap to hash
    // by (l + (p+1)/2 * (L_MAX + 1))
    int out_ptrs[(L_MAX + 1) * 2] = {0};
    int ptr = 0;
    for (int i = 0; i < irreps_o->size; i++) {
        out_ptrs[irreps_o->irreps[i].l + (irreps_o->irreps[i].p + 1) / 2 * (L_MAX + 1)] = ptr;
        ptr += irreps_o->irreps[i].c * (2 * irreps_o->irreps[i].l + 1);
    }

    int ptr1 = 0;
    for (int i1 = 0; i1 < irreps_1->size; i1++) {
        int l1 = irreps_1->irreps[i1].l;
        int c1 = irreps_1->irreps[i1].c;
        int p1 = irreps_1->irreps[i1].p;
        int l1_dim = 2 * l1 + 1;

        int ptr2 = 0;
        for (int i2 = 0; i2 < irreps_2->size; i2++) {
            int l2 = irreps_2->irreps[i2].l;
            int c2 = irreps_2->irreps[i2].c;
            int p2 = irreps_2->irreps[i2].p;
            int l2_dim = 2 * l2 + 1;

            int po = p1 * p2;
            int lo_min = abs(l1 - l2);
            int lo_max = l1 + l2;

            for (int lo = lo_min; lo <= lo_max; lo++) {
                float normalize = sqrt(2 * lo + 1);
                int out_ptr = out_ptrs[lo + (po + 1) / 2 * (L_MAX + 1)];

                for (int c1_idx = 0; c1_idx < c1; c1_idx++) {
                    for (int c2_idx = 0; c2_idx < c2; c2_idx++) {
                        for (int m1 = -l1; m1 <= l1; m1++) {
                            int idx1 = ptr1 + c1_idx * l1_dim + m1 + l1;
                            for (int m2 = -l2; m2 <= l2; m2++) {
                                int idx2 = ptr2 + c2_idx * l2_dim + m2 + l2;
                                for (int mo = -lo; mo <= lo; mo++) {
                                    float cg = clebsch_gordan(l1, l2, lo, m1, m2, mo);
                                    data_o[out_ptr + mo + lo] += (
                                        cg * data_1[idx1] * data_2[idx2] * normalize
                                    );
                                } 
                            }
                        }
                        // done writing to this irrep
                        out_ptr += (2 * lo + 1);
                    }
                }
                // done writing this chunk of irreps, shift over pointer so we
                // can write new chunks to the same irrep - this happens when
                // there is more than one path to the same irrep as long as we
                // iterate through the irreps in the same order, this should be
                // functionally equivalent to the e3nn-jax implementation 
                out_ptrs[lo + (po + 1) / 2 * (L_MAX + 1)] += (2 * lo + 1) * c1 * c2;
            }
            ptr2 += l2_dim * c2;
        }
        ptr1 += l1_dim * c1;
    }
}


void tensor_product_v2(const Irreps* irreps_1, const float* data_1, const Irreps* irreps_2, const float* data_2, const Irreps* irreps_o, float* data_o) {
    // this is the same as tensor_product above, except the inner loops over
    // m1, m2, mo are removed and replaced with lookups into the sparse
    // Clebsch-Gordan coefficients
    build_sparse_clebsch_gordan_cache();

     // Lookup table where the start of each out irrep will be in datao
    // only need to store one location per l/parity pair, it is cheap to hash by
    // by (l+ (p+1)/2 * (L_MAX + 1))
    int out_ptrs[(L_MAX + 1) * 2] = {0};
    int ptr = 0;
    for (int i = 0; i < irreps_o->size; i++) {
        out_ptrs[irreps_o->irreps[i].l + (irreps_o->irreps[i].p + 1) / 2 * (L_MAX + 1)] = ptr;
        ptr += irreps_o->irreps[i].c * (2 * irreps_o->irreps[i].l + 1);
    }

    int ptr1 = 0;
    for (int i1 = 0; i1 < irreps_1->size; i1++) {
        int l1 = irreps_1->irreps[i1].l;
        int c1 = irreps_1->irreps[i1].c;
        int p1 = irreps_1->irreps[i1].p;
        int l1_dim = 2 * l1 + 1;

        int ptr2 = 0;
        for (int i2 = 0; i2 < irreps_2->size; i2++) {
            int l2 = irreps_2->irreps[i2].l;
            int c2 = irreps_2->irreps[i2].c;
            int p2 = irreps_2->irreps[i2].p;
            int l2_dim = 2 * l2 + 1;

            int po = p1 * p2;
            int lo_min = abs(l1 - l2);
            int lo_max = l1 + l2;

            for (int lo = lo_min; lo <= lo_max; lo++) {
                float normalize = sqrt(2 * lo + 1);
                int out_ptr = out_ptrs[lo + (po + 1) / 2 * (L_MAX + 1)];
                SparseClebschGordanMatrix cg = sparse_clebsch_gordan(l1, l2, lo);

                for (int c1_idx = 0; c1_idx < c1; c1_idx++) {
                    int idx1 = ptr1 + c1_idx * l1_dim;
                    for (int c2_idx = 0; c2_idx < c2; c2_idx++) {
                        int idx2 = ptr2 + c2_idx * l2_dim;
                        for (int e=0; e<cg.size; e++) {
                            data_o[out_ptr + cg.elements[e].m3 + lo] += (
                                cg.elements[e].c 
                                * data_1[idx1 + cg.elements[e].m1 + l1] 
                                * data_2[idx2 + cg.elements[e].m2 + l2] 
                                * normalize
                            );
                        }
                        // done writing to this irrep
                        out_ptr += (2 * lo + 1);
                    }
                }
                // done writing this chunk of irreps, shift over pointer so we
                // can write new chunks to the same irrep - this happens when
                // there is more than one path to the same irrep as long as we
                // iterate through the irreps in the same order, this should be
                // functionally equivalent to the e3nn-jax implementation 
                out_ptrs[lo + (po + 1) / 2 * (L_MAX + 1)] += (2 * lo + 1) * c1 * c2;
            }
            ptr2 += l2_dim * c2;
        }
        ptr1 += l1_dim * c1;
    }
}


void tensor_product_v3(const Irreps* irreps_1, const float* data_1, const Irreps* irreps_2, const float* data_2, const Irreps* irreps_o, float* data_o) {
    // this is the same as tensor_product above, except the tensor products for
    // any l1,l2,lo are replace with a call to a precompiled version in tp.c

    // Lookup table where the start of each out irrep will be in datao
    // only need to store one location per l/parity pair, it is cheap to hash
    // by (l + (p+1)/2 * (L_MAX + 1))
    int out_ptrs[(L_MAX + 1) * 2] = {0};
    int ptr = 0;
    for (int i = 0; i < irreps_o->size; i++) {
        out_ptrs[irreps_o->irreps[i].l + (irreps_o->irreps[i].p + 1) / 2 * (L_MAX + 1)] = ptr;
        ptr += irreps_o->irreps[i].c * (2 * irreps_o->irreps[i].l + 1);
    }

    int ptr1 = 0;
    for (int i1 = 0; i1 < irreps_1->size; i1++) {
        int l1 = irreps_1->irreps[i1].l;
        int c1 = irreps_1->irreps[i1].c;
        int p1 = irreps_1->irreps[i1].p;
        int l1_dim = 2 * l1 + 1;

        int ptr2 = 0;
        for (int i2 = 0; i2 < irreps_2->size; i2++) {
            int l2 = irreps_2->irreps[i2].l;
            int c2 = irreps_2->irreps[i2].c;
            int p2 = irreps_2->irreps[i2].p;
            int l2_dim = 2 * l2 + 1;

            int po = p1 * p2;
            int lo_min = abs(l1 - l2);
            int lo_max = l1 + l2;

            for (int lo = lo_min; lo <= lo_max; lo++) {
                // float normalize = sqrt(2 * lo + 1);
                int out_ptr = out_ptrs[lo + (po + 1) / 2 * (L_MAX + 1)];
                for (int c1_idx = 0; c1_idx < c1; c1_idx++) {
                    int idx1 = ptr1 + c1_idx * l1_dim;
                    for (int c2_idx = 0; c2_idx < c2; c2_idx++) {
                        int idx2 = ptr2 + c2_idx * l2_dim;
                        tp(l1, l2, lo, data_1 + idx1, data_2 + idx2, data_o + out_ptr);
                        // done writing to this irrep
                        out_ptr += (2 * lo + 1);
                    }
                }
                // done writing this chunk of irreps, shift over pointer so we
                // can write new chunks to the same irrep - this happens when
                // there is more than one path to the same irrep as long as we
                // iterate through the irreps in the same order, this should be
                // functionally equivalent to the e3nn-jax implementation 
                out_ptrs[lo + (po + 1) / 2 * (L_MAX + 1)] += (2 * lo + 1) * c1 * c2;
            }
            ptr2 += l2_dim * c2;
        }
        ptr1 += l1_dim * c1;
    }
}


void spherical_harmonics(const Irreps* irreps, const float x, const float y, const float z, float* out) {
    int lmax = 0;
    for (int i = 0; i < irreps->size; i++) { lmax = MAX(lmax, irreps->irreps[i].l); }

    float r = sqrt(x * x + y * y + z * z);

    // NOTE: e3nn uses the following definitions for Euler angles:
    //   theta = acos(y), phi = atan2(x, z)
    float phi = atan2(x, z);

    // See note above, in the equations referenced they use x = cos(theta) which
    // is y in our coordinate system
    float x_ = (r != 0.0) ? (y / r) : 0;

    // Compute Legendre polynomials, see:
    // * Press, W.H., Teukolsky, S.A., Vetterling, W.T. and Flannery, B.P., 1992.
    //   Numerical recipes in C (pp. 252-254). New York, NY: Cambridge university
    //   press.
    // * https://github.com/chrr/libECP/blob/master/src/spherical_harmonics.c
    float* P = (float*) malloc((lmax + 1) * (lmax + 1) * sizeof(float));
    float somx2 = sqrt((1.0 - x_) * (1.0 + x_));
    float somx2m = 1.0; // store \sqrt(1 - x^2)^{1/m}
    for (int l = 0; l <= lmax; l++) {
        int m = l;
        if (l == 0) {
            P[SH_IDX(l, m)] = 1.0;
        } else {
            P[SH_IDX(l, m)] = somx2m * dfactorial(2 * m - 1);
            m = l - 1;
            P[SH_IDX(l, m)] = x_ * (2 * m + 1) * P[SH_IDX(l - 1, m)];
            if (l > 1) {
                for (m = 0; m <= l - 2; m++) {
                    P[SH_IDX(l, m)] = (
                        x_ * (2 * l - 1) * P[SH_IDX(l - 1, m)]
                        - (l + m - 1) * P[SH_IDX(l - 2, m)]
                    ) / (l - m);
                }
            }
        }
        somx2m *= somx2;
    }
    
    // component normalization
    for (int l = 0; l <= lmax; l++) {
        float norm = sqrt(2.0 * (2.0 * l + 1.0));
        // TODO: option for integral normalization, which would be:
        // float norm = sqrt((2.0 * l + 1.0) / (2.0 * M_PI));
        P[SH_IDX(l, 0)] *= norm;
        for (int m = 1; m <= l; m++) {
            P[SH_IDX(l, m)] *= sqrt(factorial(l - m) / factorial(l + m)) * norm;
        }
    }

    // precompute sin(m * phi) and cos(m * phi)
    float sin_mphi[lmax + 1];
    float cos_mphi[lmax + 1];
    if (lmax > 0) {
        sin_mphi[1] = sin(phi);
        cos_mphi[1] = cos(phi);
        for (int m = 2; m <= lmax; m++) {
            sin_mphi[m] = sin_mphi[1] * cos_mphi[m - 1] + cos_mphi[1] * sin_mphi[m - 1];
            cos_mphi[m] = cos_mphi[1] * cos_mphi[m - 1] - sin_mphi[1] * sin_mphi[m - 1];
        }
    }

    int ptr = 0;
    for (int i = 0; i < irreps->size; i++) {
        int l = irreps->irreps[i].l;
        int c = irreps->irreps[i].c;
        for (int cc = 0; cc < c; cc++) {
            for (int m = -l; m <= l; m++) {
                if (m == 0) {
                    out[ptr + l + m] = P[SH_IDX(l, m)] / sqrt(2.0);
                } else if (m < 0) {
                    out[ptr + l + m] = P[SH_IDX(l, -m)] * sin_mphi[-m];
                } else if (m > 0) {
                    out[ptr + l + m] = P[SH_IDX(l, m)] * cos_mphi[m];
                }
            }
            ptr += (2 * l + 1);
        }
    }
    free(P);
}


void linear(const Irreps* irreps_in, const float* input, const float* weight, const Irreps* irreps_out, float* out) {
    int w_ptr = 0;
    int in_ptr = 0;

    for (int i_in = 0; i_in < irreps_in->size; i_in++) {
        int out_ptr = 0;
        for (int i_out = 0; i_out < irreps_out->size; i_out++) {
            // find matching output irrep - could be done in separate loop if too costly
            if (irreps_in->irreps[i_in].l == irreps_out->irreps[i_out].l && irreps_in->irreps[i_in].p == irreps_out->irreps[i_out].p) {
                int l = irreps_in->irreps[i_in].l;
                int dim = 2 * l + 1;
                int in_c = irreps_in->irreps[i_in].c;
                int out_c = irreps_out->irreps[i_out].c;
                float norm = sqrt(1.0 / in_c);

                for (int j = 0; j < in_c; j++) {
                    for (int m = -l; m <= l; m++) {
                        for (int i = 0; i < out_c; i++) {
                            out[out_ptr + m + l + i * dim] += (
                                input[in_ptr + m + l + j * dim]
                                * weight[w_ptr + i + j * out_c]
                                * norm
                            );
                        }
                    }
                }
                // increment weight pointer to next matrix
                w_ptr += in_c * out_c;
                break;
            }
            out_ptr += (irreps_out->irreps[i_out].l * 2 + 1) * irreps_out->irreps[i_out].c;
        }
        in_ptr += (irreps_in->irreps[i_in].l * 2 + 1) * irreps_in->irreps[i_in].c;
    }
}


void concatenate(const Irreps* irreps_1, const float* data_1, const Irreps* irreps_2, const float* data_2, float* data_o) {
    assert(irreps_is_sorted(irreps_1));
    assert(irreps_is_sorted(irreps_2));
    int inc_1 = 0, inc_2 = 0, inc_o = 0;
    int i1 = 0, i2 = 0;
    while (i1 < irreps_1->size || i2 < irreps_2->size) {
        if (
            (i1 < irreps_1->size) &&
            (i2 >= irreps_2->size || irrep_compare(&irreps_1->irreps[i1], &irreps_2->irreps[i2]) <= 0)) {
            int dim = irrep_dim(&irreps_1->irreps[i1]);
            memcpy(data_o + inc_o, data_1 + inc_1, sizeof(float) * dim);
            inc_1 += dim;
            inc_o += dim;
            i1++;
        } else {
            int dim = irrep_dim(&irreps_2->irreps[i2]);
            memcpy(data_o + inc_o, data_2 + inc_2, sizeof(float) * dim);
            inc_2 += dim;
            inc_o += dim;
            i2++;
        }
    }
}




void irreps_to_out(float x, float y, float z, float* out_feature)
{

    float node_position_sh[9] = { 0 };
    // x = 1.0; y = 2.0; z = 3.0;
    Irreps* node_irreps = irreps_create("1x0e + 1x1o + 1x2e");
    spherical_harmonics(node_irreps, x,y,z, node_position_sh);

    printf("sh ["); for (int i = 0; i < 9; i++){ printf("%.2f, ", node_position_sh[i]); } printf("]\n");
    // irreps_free(node_irreps);

    float neighbor_feature[] = { 7, 8, 9 };
    float product[27] = { 0 };
    Irreps* node_sh_irreps = irreps_create("1x0e + 1x1o + 1x2e");
    Irreps* neighbor_feature_irreps = irreps_create("1x1e");
    Irreps* product_irreps = irreps_create("1x0o + 1x1o + 2x1e + 1x2e + 1x2o + 1x3e");
    tensor_product(node_sh_irreps, node_position_sh, 
                   neighbor_feature_irreps, neighbor_feature, 
                   product_irreps, product);

    printf("product ["); for (int i = 0; i < 27; i++){ printf("%.2f, ", product[i]); } printf("]\n");
    // irreps_free(node_sh_irreps);
    // irreps_free(neighbor_feature_irreps);

    float weights[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    //                [ 1 x 1 weight] [1 x 1 weight] [2 x 2 weight] [1 x 1 weight] [1 x 1 weight] [ 1 x 1 weight]
    float output[27] = { 0 };
    Irreps* output_irreps = irreps_create("1x0o + 1x1o + 2x1e + 1x2e + 1x2o + 1x3e");
    linear(product_irreps,
           product,
           weights,
           output_irreps,
           output);

    printf("output ["); for (int i = 0; i < 27; i++) { printf("%.2f, ", output[i]); } printf("]\n");
    for (int i = 0; i < 27; i++) { out_feature[i] = output[i]; }
    // irreps_free(product_irreps);
    // irreps_free(output_irreps);

}

int enzyme_dup;

void __enzyme_autodiff(void (*) (float, float, float, float*), 
                        int, float, float,
                        int, float, float, 
                        int, float, float,
                        int, float*, float*);

int main(){
    float out_feature[27] = { 0 };
    float out_grad[27] = { 1 };
    float x = 1.0, y = 2.0, z = 3.0;
    float x_grad = 0.0, y_grad = 0.0, z_grad = 0.0;
    irreps_to_out(1.0, 2.0, 3.0, out_feature);
    printf("out_feature ["); for (int i = 0; i < 27; i++) { printf("%.2f, ", out_feature[i]); } printf("]\n");
    __enzyme_autodiff(irreps_to_out, enzyme_dup, x, x_grad, 
                        enzyme_dup, y, y_grad, 
                        enzyme_dup, z, z_grad, 
                        enzyme_dup, out_feature, out_grad);
    printf("x_grad: %.2f, y_grad: %.2f, z_grad: %.2f\n", x_grad, y_grad, z_grad);
    return 0;
}