/**
 * @file args.h
 * @author Cassio E. dos Santos Jr.
 * @date may/2013
 */
#ifndef _ARGS_H_
#define _ARGS_H_


#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>


#ifdef __cplusplus


#include <vector>
#include <string>
#include <sstream>


extern "C" {
#endif


int c_argc;
char **c_argv;
char *c_shortopts = NULL;
struct option *c_longopts = NULL;


/**
 * @param name: Long name (--input).
 * @param option: Short name character (-i).
 * @param argument: Whether 'no_argument', 'required_argument'
 *                  or 'optional_argument'.
 * @param description: Brief description of the argument
 *                     that appears when help is called
 */
#define c_argument(name, option, argument, description) \
    {{name, argument, 0, option}, description},


struct c_option {
    struct option opt;
    const char *desc;
};


const char* c_get_description();


/** 
 * @param desc: Brief description of the program. Appears when help is called.
 */
#define c_set_description(desc) \
    const char* c_get_description() { \
        static const char _desc[] = desc; \
        return _desc; \
    }


const struct c_option* c_get_arguments();


/**
 * @param one or more c_arguments.
 */
#define c_set_arguments(...) \
    const struct c_option* c_get_arguments() { \
        static const struct c_option args[] = { __VA_ARGS__ \
            {{"help", no_argument, 0, 'h'}, "print this help"}, {{0, 0, 0, 0}, 0} }; \
        static struct option _c_longopts[sizeof(args)/sizeof(struct c_option)]; \
        static char _c_shortopts[3*sizeof(args)/sizeof(struct c_option)]; \
        if(!c_longopts) c_longopts = _c_longopts; \
        if(!c_shortopts) c_shortopts = _c_shortopts; \
        return args; \
    }

    
/** @brief Initialize the library. Required to be called with main's arguments.
 */
void c_init(int argc, char** argv) {
    
    int i;
    int c;
    const struct c_option *c_opts;

    c_argc = argc;
    c_argv = argv;
    
    c_opts = c_get_arguments();
    for( i = 0, c = 0; c_opts[i].desc; ++i ) {
        // set long opts
        c_longopts[i] = c_opts[i].opt;

        // set short opts
        c_shortopts[c++] = c_opts[i].opt.val;
        if(c_opts[i].opt.has_arg == required_argument)
            c_shortopts[c++] = ':';
        if(c_opts[i].opt.has_arg == optional_argument) {
            c_shortopts[c++] = ':';
            c_shortopts[c++] = ':';
        }
    }
    c_shortopts[c] = '\0';
}


void c_usage() {

    int i;
    const struct c_option* opts;

    printf("%s\n", c_get_description());

    opts = c_get_arguments();
    for( i = 0; opts[i].desc; ++i )

        printf("\t-%c, --%s:\t%s\n",
            opts[i].opt.val,
            opts[i].opt.name,
            opts[i].desc
        );
}


/**
 * @param opt: return a string pointer to the short name argument opt.
 * 
 * @return If no argument is informed, then NULL is returned.
 *         If an optional argument is not informed, "" is returned.
 *         If the option was informed but no argument is required, "" is returned.
 */
const char* c_get(const char opt) {

    char c;
    extern char *optarg;
    extern int optind;
    int longopt_index;

    optind = 1;
    longopt_index = -1;
    
    while((c = getopt_long(c_argc, c_argv, c_shortopts, c_longopts, &longopt_index)) != -1)

        if(c == opt) {
            // set longopt_index in case long name was not informed
            if(longopt_index)
                for(longopt_index = 0; c_longopts[longopt_index].name; ++longopt_index )
                    if(c_longopts[longopt_index].val == c)
                        break;
            // return empty string case no argument is required
            if(c_longopts[longopt_index].has_arg == no_argument)
                return "";
            else if(c_longopts[longopt_index].has_arg == optional_argument
                    && optarg == NULL)
            {
                return "";
            }
            else
                return optarg;
        }
        else if(c == '?' || c == 'h') {
            c_usage();
            exit(EXIT_SUCCESS);
        }

    return NULL;
}


void c_print() {
    
    int i;
    const struct c_option* opts;

    opts = c_get_arguments();
    for( i = 0; opts[i].desc; ++i ) {

        if(opts[i].opt.val == 'h')
            continue;

        const char *val = c_get(opts[i].opt.val);

        printf("-%c, --%s\t=\t%s\n",
            opts[i].opt.val,
            opts[i].opt.name,
            val ? val : ""
        );
    }
}


#ifdef __cplusplus

}  // extern C


class Args{

    
public:

    Args(int argc, char** argv) {
        c_init(argc, argv);
    }

    bool get(const char opt) {
        return c_get(opt);
    }

    std::string& get(std::string &var, const char opt) {
        const char* c = c_get(opt);
        if(c) var = c;
        return var;
    }

    void print() {
        c_print();
    }
    
    std::vector<std::string>& get(std::vector<std::string> &var, const char opt) {

        extern int optind;
        var.push_back(c_get(opt));

        while((optind < c_argc) && (c_argv[optind][0] != '-'))
            var.push_back(c_argv[optind++]);
        return var;
    }

    template<class T>
    T& get(T &var, const char opt) {

        const char* val = c_get(opt);        
        if(!val)
            return var;

        std::stringstream parser(val);
        parser >> var;
        return var;
    }
};

#endif

#endif