fallback_intensity() = SImageND(IntensityPixel{N0f8}.(ones(N0f8, size(sample_img))))
fallback_binary() = SImageND(BinaryPixel{Bool}.(ones(Bool, size(sample_img))))
fallback_segment() = SImageND(SegmentPixel{Int}.(ones(Int, size(sample_img))))

image_intensity = UTCGP.get_image2Dintensity_factory_bundles();
image_binary = UTCGP.get_image2Dbinary_factory_bundles();
image_segment = UTCGP.get_image2Dsegment_factory_bundles();
Type2Dimg_intensity = typeof(fallback_intensity())
Type2Dimg_binary = typeof(fallback_binary())
Type2Dimg_segment = typeof(fallback_segment())
@show Type2Dimg_intensity Type2Dimg_binary Type2Dimg_segment

# push!(image2D, UTCGP.experimental_bundle_image2D_mask_factory) # TODO

function set_bundle_casters!(bundles, caster)
    for factory_bundle in bundles
        for (i, wrapper) in enumerate(factory_bundle)
            wrapper.caster = caster
        end
    end
    return
end

for (factories, fallback, typeimg) in zip(
        [image_intensity, image_binary, image_segment],
        [fallback_intensity, fallback_binary, fallback_segment],
        [Type2Dimg_intensity, Type2Dimg_binary, Type2Dimg_segment],
    )
    for factory_bundle in factories
        for (i, wrapper) in enumerate(factory_bundle)
            fn = wrapper.fn(typeimg) # specialize
            wrapper.fallback = fallback
            factory_bundle.functions[i] =
                UTCGP.FunctionWrapper(fn, wrapper.name, wrapper.caster, wrapper.fallback)
        end
    end
end
