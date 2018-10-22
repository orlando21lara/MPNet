// THIS IS AN AUTOMATICALLY GENERATED FILE.  DO NOT MODIFY
// BY HAND!!
//
// Generated by lcm-gen

#include <string.h>
#include "lcmtypes/lcmtypes_edge_t.h"

static int __lcmtypes_edge_t_hash_computed;
static uint64_t __lcmtypes_edge_t_hash;

uint64_t __lcmtypes_edge_t_hash_recursive(const __lcm_hash_ptr *p)
{
    const __lcm_hash_ptr *fp;
    for (fp = p; fp != NULL; fp = fp->parent)
        if (fp->v == __lcmtypes_edge_t_get_hash)
            return 0;

    __lcm_hash_ptr cp;
    cp.parent =  p;
    cp.v = (void*)__lcmtypes_edge_t_get_hash;
    (void) cp;

    uint64_t hash = (uint64_t)0x1fae492d71eedf94LL
         + __lcmtypes_vertex_t_hash_recursive(&cp)
         + __lcmtypes_vertex_t_hash_recursive(&cp)
         + __lcmtypes_trajectory_t_hash_recursive(&cp)
        ;

    return (hash<<1) + ((hash>>63)&1);
}

int64_t __lcmtypes_edge_t_get_hash(void)
{
    if (!__lcmtypes_edge_t_hash_computed) {
        __lcmtypes_edge_t_hash = (int64_t)__lcmtypes_edge_t_hash_recursive(NULL);
        __lcmtypes_edge_t_hash_computed = 1;
    }

    return __lcmtypes_edge_t_hash;
}

int __lcmtypes_edge_t_encode_array(void *buf, int offset, int maxlen, const lcmtypes_edge_t *p, int elements)
{
    int pos = 0, element;
    int thislen;

    for (element = 0; element < elements; element++) {

        thislen = __lcmtypes_vertex_t_encode_array(buf, offset + pos, maxlen - pos, &(p[element].vertex_src), 1);
        if (thislen < 0) return thislen; else pos += thislen;

        thislen = __lcmtypes_vertex_t_encode_array(buf, offset + pos, maxlen - pos, &(p[element].vertex_dst), 1);
        if (thislen < 0) return thislen; else pos += thislen;

        thislen = __lcmtypes_trajectory_t_encode_array(buf, offset + pos, maxlen - pos, &(p[element].trajectory), 1);
        if (thislen < 0) return thislen; else pos += thislen;

    }
    return pos;
}

int lcmtypes_edge_t_encode(void *buf, int offset, int maxlen, const lcmtypes_edge_t *p)
{
    int pos = 0, thislen;
    int64_t hash = __lcmtypes_edge_t_get_hash();

    thislen = __int64_t_encode_array(buf, offset + pos, maxlen - pos, &hash, 1);
    if (thislen < 0) return thislen; else pos += thislen;

    thislen = __lcmtypes_edge_t_encode_array(buf, offset + pos, maxlen - pos, p, 1);
    if (thislen < 0) return thislen; else pos += thislen;

    return pos;
}

int __lcmtypes_edge_t_encoded_array_size(const lcmtypes_edge_t *p, int elements)
{
    int size = 0, element;
    for (element = 0; element < elements; element++) {

        size += __lcmtypes_vertex_t_encoded_array_size(&(p[element].vertex_src), 1);

        size += __lcmtypes_vertex_t_encoded_array_size(&(p[element].vertex_dst), 1);

        size += __lcmtypes_trajectory_t_encoded_array_size(&(p[element].trajectory), 1);

    }
    return size;
}

int lcmtypes_edge_t_encoded_size(const lcmtypes_edge_t *p)
{
    return 8 + __lcmtypes_edge_t_encoded_array_size(p, 1);
}

int __lcmtypes_edge_t_decode_array(const void *buf, int offset, int maxlen, lcmtypes_edge_t *p, int elements)
{
    int pos = 0, thislen, element;

    for (element = 0; element < elements; element++) {

        thislen = __lcmtypes_vertex_t_decode_array(buf, offset + pos, maxlen - pos, &(p[element].vertex_src), 1);
        if (thislen < 0) return thislen; else pos += thislen;

        thislen = __lcmtypes_vertex_t_decode_array(buf, offset + pos, maxlen - pos, &(p[element].vertex_dst), 1);
        if (thislen < 0) return thislen; else pos += thislen;

        thislen = __lcmtypes_trajectory_t_decode_array(buf, offset + pos, maxlen - pos, &(p[element].trajectory), 1);
        if (thislen < 0) return thislen; else pos += thislen;

    }
    return pos;
}

int __lcmtypes_edge_t_decode_array_cleanup(lcmtypes_edge_t *p, int elements)
{
    int element;
    for (element = 0; element < elements; element++) {

        __lcmtypes_vertex_t_decode_array_cleanup(&(p[element].vertex_src), 1);

        __lcmtypes_vertex_t_decode_array_cleanup(&(p[element].vertex_dst), 1);

        __lcmtypes_trajectory_t_decode_array_cleanup(&(p[element].trajectory), 1);

    }
    return 0;
}

int lcmtypes_edge_t_decode(const void *buf, int offset, int maxlen, lcmtypes_edge_t *p)
{
    int pos = 0, thislen;
    int64_t hash = __lcmtypes_edge_t_get_hash();

    int64_t this_hash;
    thislen = __int64_t_decode_array(buf, offset + pos, maxlen - pos, &this_hash, 1);
    if (thislen < 0) return thislen; else pos += thislen;
    if (this_hash != hash) return -1;

    thislen = __lcmtypes_edge_t_decode_array(buf, offset + pos, maxlen - pos, p, 1);
    if (thislen < 0) return thislen; else pos += thislen;

    return pos;
}

int lcmtypes_edge_t_decode_cleanup(lcmtypes_edge_t *p)
{
    return __lcmtypes_edge_t_decode_array_cleanup(p, 1);
}

int __lcmtypes_edge_t_clone_array(const lcmtypes_edge_t *p, lcmtypes_edge_t *q, int elements)
{
    int element;
    for (element = 0; element < elements; element++) {

        __lcmtypes_vertex_t_clone_array(&(p[element].vertex_src), &(q[element].vertex_src), 1);

        __lcmtypes_vertex_t_clone_array(&(p[element].vertex_dst), &(q[element].vertex_dst), 1);

        __lcmtypes_trajectory_t_clone_array(&(p[element].trajectory), &(q[element].trajectory), 1);

    }
    return 0;
}

lcmtypes_edge_t *lcmtypes_edge_t_copy(const lcmtypes_edge_t *p)
{
    lcmtypes_edge_t *q = (lcmtypes_edge_t*) malloc(sizeof(lcmtypes_edge_t));
    __lcmtypes_edge_t_clone_array(p, q, 1);
    return q;
}

void lcmtypes_edge_t_destroy(lcmtypes_edge_t *p)
{
    __lcmtypes_edge_t_decode_array_cleanup(p, 1);
    free(p);
}

int lcmtypes_edge_t_publish(lcm_t *lc, const char *channel, const lcmtypes_edge_t *p)
{
      int max_data_size = lcmtypes_edge_t_encoded_size (p);
      uint8_t *buf = (uint8_t*) malloc (max_data_size);
      if (!buf) return -1;
      int data_size = lcmtypes_edge_t_encode (buf, 0, max_data_size, p);
      if (data_size < 0) {
          free (buf);
          return data_size;
      }
      int status = lcm_publish (lc, channel, buf, data_size);
      free (buf);
      return status;
}

struct _lcmtypes_edge_t_subscription_t {
    lcmtypes_edge_t_handler_t user_handler;
    void *userdata;
    lcm_subscription_t *lc_h;
};
static
void lcmtypes_edge_t_handler_stub (const lcm_recv_buf_t *rbuf,
                            const char *channel, void *userdata)
{
    int status;
    lcmtypes_edge_t p;
    memset(&p, 0, sizeof(lcmtypes_edge_t));
    status = lcmtypes_edge_t_decode (rbuf->data, 0, rbuf->data_size, &p);
    if (status < 0) {
        fprintf (stderr, "error %d decoding lcmtypes_edge_t!!!\n", status);
        return;
    }

    lcmtypes_edge_t_subscription_t *h = (lcmtypes_edge_t_subscription_t*) userdata;
    h->user_handler (rbuf, channel, &p, h->userdata);

    lcmtypes_edge_t_decode_cleanup (&p);
}

lcmtypes_edge_t_subscription_t* lcmtypes_edge_t_subscribe (lcm_t *lcm,
                    const char *channel,
                    lcmtypes_edge_t_handler_t f, void *userdata)
{
    lcmtypes_edge_t_subscription_t *n = (lcmtypes_edge_t_subscription_t*)
                       malloc(sizeof(lcmtypes_edge_t_subscription_t));
    n->user_handler = f;
    n->userdata = userdata;
    n->lc_h = lcm_subscribe (lcm, channel,
                                 lcmtypes_edge_t_handler_stub, n);
    if (n->lc_h == NULL) {
        fprintf (stderr,"couldn't reg lcmtypes_edge_t LCM handler!\n");
        free (n);
        return NULL;
    }
    return n;
}

int lcmtypes_edge_t_subscription_set_queue_capacity (lcmtypes_edge_t_subscription_t* subs,
                              int num_messages)
{
    return lcm_subscription_set_queue_capacity (subs->lc_h, num_messages);
}

int lcmtypes_edge_t_unsubscribe(lcm_t *lcm, lcmtypes_edge_t_subscription_t* hid)
{
    int status = lcm_unsubscribe (lcm, hid->lc_h);
    if (0 != status) {
        fprintf(stderr,
           "couldn't unsubscribe lcmtypes_edge_t_handler %p!\n", hid);
        return -1;
    }
    free (hid);
    return 0;
}

