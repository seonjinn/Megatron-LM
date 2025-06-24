from megatron.core.models.vision.clip_vit_model import get_num_image_embeddings


class ImageEmbeddings:
    def __init__(
            self,
            img_h: int,
            img_w: int,
            patch_dim: int,
            vision_model_type: str,
            disable_vision_class_token: bool,
            class_token_len: int,
            pixel_shuffle: bool,
            use_tile_tags: bool,
            max_num_tiles: int,
            tokenizer_type: str,
            use_image_break_token: bool,
            conv_merging: bool,
            dynamic: bool,
    ):
        self.img_h = img_h
        self.img_w = img_w
        self.patch_dim = patch_dim
        self.vision_model_type = vision_model_type
        self.disable_vision_class_token = disable_vision_class_token
        self.class_token_len = class_token_len
        self.pixel_shuffle = pixel_shuffle
        self.use_tile_tags = use_tile_tags
        self.max_num_tiles = max_num_tiles
        self.tokenizer_type = tokenizer_type
        self.use_image_break_token = use_image_break_token
        self.conv_merging = conv_merging
        self.dynamic = dynamic
        if not dynamic:
            self.num_image_embeddings_per_tile = get_num_image_embeddings(
                img_h=self.img_h,
                img_w=self.img_w,
                patch_dim=self.patch_dim,
                vision_model_type=self.vision_model_type,
                disable_vision_class_token=self.disable_vision_class_token,
                class_token_len=self.class_token_len,
                pixel_shuffle=self.pixel_shuffle,
                use_tile_tags=self.use_tile_tags,
                max_num_tiles=self.max_num_tiles,
                tokenizer_type=self.tokenizer_type,
                use_image_break_token=self.use_image_break_token,
                conv_merging=self.conv_merging,
            )

    def __call__(self, img_sizes: list[tuple[int, int]]) -> int:
        if self.dynamic:
            num_image_embeddings = 0
            for img_size in img_sizes:
                num_image_embeddings += get_num_image_embeddings(
                    img_h=img_size[1],
                    img_w=img_size[0],
                    patch_dim=self.patch_dim,
                    vision_model_type=self.vision_model_type,
                    disable_vision_class_token=self.disable_vision_class_token,
                    class_token_len=self.class_token_len,
                    pixel_shuffle=self.pixel_shuffle,
                    use_tile_tags=self.use_tile_tags,
                    max_num_tiles=self.max_num_tiles,
                    tokenizer_type=self.tokenizer_type,
                    use_image_break_token=self.use_image_break_token,
                    conv_merging=self.conv_merging,
                )
            return num_image_embeddings
        else:
            return self.num_image_embeddings_per_tile * len(img_sizes)
