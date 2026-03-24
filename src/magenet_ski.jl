function define_common_image_functions(sample_img)
    global float_caster2
    @everywhere float_caster2 = function (n)
        if isnan(n) || isinf(n) || isnothing(n)
            return 0.0
        else
            clamp(convert(Float64, n), UTCGP.MIN_FLOAT[], UTCGP.MAX_FLOAT[])
        end
    end
    global ret_0
    @everywhere ret_0 = () -> 0.0
    global Type2Dimg
    return @everywhere Type2Dimg = typeof($sample_img)
end

function setup_addprocs!(nt)
    return addprocs(nt, exeflags = ["--threads=1"])
end

function setup_skimage_distributed(Type2Dimg)
    skimage_factories = [bundle_float_skimagemeasure]
    skimage_factories = [deepcopy(b) for b in skimage_factories]
    for factory_bundle in skimage_factories
        for (i, wrapper) in enumerate(factory_bundle)
            fn = wrapper.fn(Type2Dimg) # specialize
            wrapper.fallback = ret_0
            wrapper.caster = float_caster2
            factory_bundle.functions[i] =
                UTCGP.FunctionWrapper(fn, wrapper.name, wrapper.caster, wrapper.fallback)
        end
    end
    return skimage_factories
end

# LEGACY. When functions where evaled in MAGE_SKIMAGE_MEASURE
# function setup_skimage_distributed(Type2Dimg)
#     return @everywhere begin
#         skimage_factories = [bundle_float_skimagemeasure]
#         skimage_factories = [deepcopy(b) for b in skimage_factories]
#         for factory_bundle in skimage_factories
#             for (i, wrapper) in enumerate(factory_bundle)
#                 fn = wrapper.fn($Type2Dimg) # specialize
#                 wrapper.fallback = ret_0
#                 wrapper.caster = float_caster2
#                 factory_bundle.functions[i] =
#                     UTCGP.FunctionWrapper(fn, wrapper.name, wrapper.caster, wrapper.fallback)

#                 # specific_functions = typeof(fn).parameters[1]
#                 # function (dp::ManualDispatcher{specific_functions})(inputs::Vararg{Any})
#                 #     tt = typeof.(inputs)
#                 #     fn = UTCGP._which_fn_in_manual_dispatcher(dp, tt)
#                 #     if isnothing(fn) || !(fn isa Function)
#                 #         msg = MethodError(dp, 1)
#                 #         throw(msg)
#                 #     else
#                 #         fn_sure = identity(fn)
#                 #         @fetch begin
#                 #             fn_sure(inputs...)
#                 #         end
#                 #     end
#                 # end
#             end
#         end
#     end
# end
