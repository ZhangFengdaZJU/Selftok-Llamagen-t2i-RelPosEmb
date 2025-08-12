import sys 
sys.path.append('./')

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)     # disable default parameter init for faster speed
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)  # disable default parameter init for faster speed
from torchvision.utils import save_image

import os
import time
import argparse
from tokenizer.tokenizer_image.vq_model import VQ_models
from language.t5 import T5Embedder
from autoregressive.models.gpt import GPT_models
from autoregressive.models.generate import generate
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from mimogpt.infer.SelftokPipeline import SelftokPipeline
from mimogpt.infer.SelftokPipeline import NormalizeToTensor
from mimogpt.infer.infer_utils import parse_args_from_yaml
from torchvision.utils import save_image


def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # create and load model
    cfg = parse_args_from_yaml(args.yml_path)
    vq_model = SelftokPipeline(cfg=cfg, ckpt_path=args.vq_ckpt, sd3_path=args.sd3_pretrained, datasize=args.image_size, device=device)


    # create and load gpt model
    precision = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.precision]
    latent_size = args.image_size // args.downsample_size
    gpt_model = GPT_models[args.gpt_model](
        block_size=1536,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
    ).to(device=device, dtype=precision)

    checkpoint = torch.load(args.gpt_ckpt, map_location="cpu", weights_only=False)
 
    if "model" in checkpoint:  # ddp
        model_weight = checkpoint["model"]
    elif "module" in checkpoint: # deepspeed
        model_weight = checkpoint["module"]
    elif "state_dict" in checkpoint:
        model_weight = checkpoint["state_dict"]
    else:
        raise Exception("please check model weight")
    gpt_model.load_state_dict(model_weight, strict=True)
    gpt_model.eval()
    del checkpoint
    print(f"gpt model is loaded")

    if args.compile:
        print(f"compiling the model...")
        gpt_model = torch.compile(
            gpt_model,
            mode="reduce-overhead",
            fullgraph=True
        ) # requires PyTorch 2.0 (optional)
    else:
        print(f"no need to compile model in demo") 
    
    assert os.path.exists(args.t5_path)
    t5_model = T5Embedder(
        device=device, 
        local_cache=True, 
        cache_dir=args.t5_path, 
        dir_or_name=args.t5_model_type,
        torch_dtype=precision,
        model_max_length=args.t5_feature_max_len,
    )

    # prompts = [
    #     'The image is a photograph of a vintage metal tool, likely a type of plane or smoothing tool, placed on a plain, light-colored surface. The overall atmosphere is minimalist and utilitarian, focusing on the object itself without any additional context or distractions.\n\nThe tool has a metallic body with a silver-gray finish, showing signs of wear and age. It features a wooden knob on the left side, which is round and polished, contrasting with the metal. The body of the tool is elongated and slightly curved, with a flat base that allows it to rest on the surface. \n\nOn the right side, there is a wooden component that appears to be a part of the tool\'s mechanism, possibly a lever or adjustment piece. This wooden part is darker in color, with a reddish-brown hue, and is positioned at an angle, adding a dynamic element to the composition.\n\nA small rectangular label with the number "4212" is affixed to the metal body near the wooden knob. The label is brown with white numerals, providing a subtle contrast against the metallic surface.\n\nThe composition style is straightforward and focused, with the tool centrally positioned in the frame. The background is plain and out of focus, ensuring that the viewer\'s attention remains on the tool. The lighting is even, with no harsh shadows, highlighting the tool\'s details and texture.\n\nThe dominant color tones are muted, with shades of silver, gray, and brown. The image has a balanced contrast and moderate saturation, emphasizing the tool\'s features without overwhelming the viewer. The background is neutral and lacks any discernible blur, maintaining a clean and simple presentation. \n\nOverall, the image captures the essence of a well-used, vintage tool, showcasing its design and craftsmanship in a clear and focused manner.',
    #     'The image features a collage of three comic book covers laid out on a textured surface. The overall atmosphere is vibrant and dynamic, with a focus on action and storytelling typical of comic book art. The composition style is typical of promotional comic book displays, showcasing the artwork and titles prominently.\n\n1. **Top Left Comic: "Punisher War Journal"**\n   - **Color and Visual Characteristics:** The cover is dominated by dark tones with a striking contrast between black and red. The background is mostly black, with a red, fiery explosion effect.\n   - **Characters and Details:** The central figure is a menacing character with a wide, blood-curdling grin, painted in shades of red and orange. Below him, a shadowy figure in a black cloak wields a sword, standing amidst a scene of destruction with skulls scattered around.\n   - **Text:** The title "Punisher War Journal" is prominently displayed at the top in bold, white letters. The Marvel logo is visible in the top right corner.\n\n2. **Top Right Comic: "Cable"**\n   - **Color and Visual Characteristics:** This cover features a mix of dark and vibrant colors. The background is a gradient of deep red and purple, with dynamic action scenes.\n   - **Characters and Details:** The cover shows a central character, Cable, in a heroic pose, wielding a gun. Behind him, two other characters are engaged in combat, adding to the sense of action.\n   - **Text:** The title "Cable" is at the top in bold white letters. The text "Now with 400 pages of action!" is highlighted in a red box. The Marvel logo is also present.\n\n3. **Bottom Comic: "House of Mystery"**\n   - **Color and Visual Characteristics:** The cover is more subdued with earthy tones. The background features a muted, sepia-like color palette.\n   - **Characters and Details:** The main character is a woman with long, reddish-brown hair, smiling and holding a violin. She is dressed in a blue shirt and brown vest, with a belt around her waist.\n   - **Text:** The title "House of Mystery" is prominently displayed in the center. Below it, "Stories by Roger Avary" is noted. The text "Love Stories for Dead People" is highlighted in a red box. The issue number "Verdico" is also visible.\n\n**Composition and Framing:**\nThe three covers are arranged in a triangular layout, with the "Punisher',
    #     "The image is a photograph capturing a small bird standing on a bed of gray pebbles. The atmosphere is natural and serene, with a focus on the bird and its immediate surroundings.\n\nThe bird is positioned slightly to the left of the center from the viewer's perspective. It has a predominantly brown body with a darker gray head and a lighter, tan-colored belly. Its wings are a mix of brown and gray, with subtle patterning visible. The bird's beak is short and pinkish, and its legs are thin and reddish-brown, blending slightly with the pebbles.\n\nThe pebbles are small and uniformly gray, creating a textured background that contrasts with the bird's colors. They are scattered randomly, covering the entire ground in the image. There are no other objects or characters in the image, and the background is blurred, drawing attention to the bird.\n\nThe composition style is straightforward, with the bird as the focal point. The framing is centered on the bird, capturing it from a slightly elevated angle. The image has a natural color tone, with muted colors and moderate contrast. The pebbles and the bird's feathers are sharp, while the background is softly blurred, enhancing the focus on the bird.\n\nOverall, the image highlights the simplicity and beauty of a small bird in its natural habitat, with a balanced use of color and texture.",
    #     'The image is a photograph capturing a close-up view of a zucchini plant growing in a container. The atmosphere is vibrant and lush, with a focus on the plant\'s healthy green foliage and elongated zucchini fruits.\n\n**Objects and Features:**\n1. **Zucchini Plant:** The plant has several long, green zucchini fruits growing from it. The fruits are prominently displayed, with one large zucchini curving across the image and another smaller one positioned vertically. The leaves are broad and green, with visible veins and a slightly glossy texture.\n2. **Container:** The plant is growing in a white container that has a green bag or liner inside it. The liner has illustrations of various vegetables, including strawberries and eggplants, along with text. The text on the liner reads "Eggplants" and "Strawberries," with additional smaller text that is not fully legible.\n3. **Purple Bowl:** A small purple bowl is nestled among the plant\'s leaves and fruits, adding a pop of color to the scene.\n4. **Background:** The background is slightly blurred, suggesting a greenhouse or garden setting with natural light filtering through. The green frame of a window or structure is visible, enhancing the outdoor feel.\n\n**Composition and Style:**\n- The composition is dynamic, with the zucchini fruits and leaves creating a sense of movement and growth. The perspective is slightly tilted, adding to the natural, candid feel of the photograph.\n- The framing focuses on the plant, with the background softly blurred to keep attention on the main subject.\n\n**Color Tones and Grading:**\n- The image has high saturation and brightness, emphasizing the vivid green of the plant and the colorful illustrations on the liner.\n- The contrast is moderate, ensuring that all elements are distinguishable without appearing overly harsh.\n\n**Relative Position and Interaction:**\n- The zucchini fruits are prominently in the foreground, with one extending diagonally across the image and another positioned vertically near the center.\n- The leaves spread out, partially covering the liner and bowl, creating a layered effect.\n- The purple bowl is situated near the center, partially hidden by the leaves and fruits.\n\n**Text:**\n- The visible text on the liner includes "Eggplants" and "Strawberries," along with smaller, less legible text.\n\nOverall, the image captures a thriving zucchini plant in a colorful and well-lit setting, emphasizing the freshness and vitality of the produce.'
    # ]

    prompts = [
            "The image is a photograph capturing a joyful scene in a lush, green outdoor setting. The atmosphere is bright and cheerful, with a focus on natural beauty and warmth. In the center of the image, there is a golden retriever lying on its stomach in a grassy field. The dog has a rich, golden coat that appears soft and well-groomed. Its tongue is playfully sticking out, and it has a wide, happy smile, suggesting a sense of contentment and joy. The dog is wearing a black harness, which is positioned snugly around its chest and back. The grass is vibrant green, dotted with small white flowers, adding a delicate touch to the scene. The flowers are scattered randomly across the field, creating a natural and unstructured pattern. The background is slightly blurred, indicating a shallow depth of field, which helps to keep the focus on the dog. This background blur also suggests a larger, open space, possibly a park or garden. The composition style is centered, with the dog positioned slightly off-center to the left, drawing the viewer's eye directly to its happy expression. The framing is natural, capturing the dog in its environment without any artificial boundaries. The dominant color tones are green from the grass and white from the flowers, with the golden hue of the dog's fur providing a warm contrast. The image has high saturation and brightness, enhancing the cheerful and lively mood. The sharpness is clear, particularly on the dog, while the background blur adds depth and focus to the subject. Overall, the image exudes warmth and happiness, with the golden retriever as the central, joyful character in a serene, natural setting.",
            "The image is a photograph capturing a serene and heartwarming scene. It features a small, white puppy lying on a wooden deck. The puppy has a soft, fluffy coat and large, expressive eyes that convey a sense of curiosity and innocence. Its ears are floppy and hang down on either side of its head, and it wears a green collar around its neck, adding a touch of color to its predominantly white fur. The puppy is positioned centrally in the frame, facing the camera directly, which draws the viewer's attention immediately to its adorable features. Its front paws are stretched out in front of it, while its hind legs are tucked under its body, giving it a relaxed and comfortable appearance. The background is softly blurred, creating a bokeh effect that highlights the puppy as the focal point. The blurred greenery suggests an outdoor setting, likely a garden or park, contributing to a peaceful and natural atmosphere. The lighting is bright and natural, indicating that the photo was taken during the day, possibly on a sunny afternoon. The composition style is simple yet effective, with the puppy placed centrally to create a balanced and engaging image. The color tones are warm and inviting, with a high saturation that enhances the puppy's white fur and the green collar. The background blur and the natural lighting add depth and a sense of tranquility to the scene. Overall, the image exudes warmth and cuteness, capturing a moment of innocent curiosity in a serene outdoor setting.",
            "The image is a photograph of a cheerful golden retriever puppy. The atmosphere is warm and inviting, with a soft, bright background that enhances the puppy's golden fur. The puppy is positioned centrally in the frame, sitting on a dark surface, likely a table or bench, which contrasts with its light fur. The puppy has a fluffy, golden coat with a slightly lighter shade on its chest and face. Its ears are floppy and hang down on either side of its head. The puppy's eyes are bright and expressive, and its mouth is open in a joyful smile, revealing its small pink tongue. The puppy's overall appearance is playful and endearing, exuding happiness and energy. The composition style is straightforward and focused, with the puppy being the sole subject of the image. The background is plain and light, ensuring that the puppy stands out prominently. The lighting is soft and even, highlighting the puppy's fur and giving it a gentle glow. The image has high saturation and brightness, with a slight blur in the background to keep the attention on the puppy. There is no discernible text in the image. The dominant color tones are warm, with golden and beige hues dominating the scene. The contrast between the puppy's fur and the dark surface it sits on adds depth to the image. The overall effect is a heartwarming and visually appealing photograph that captures the puppy's joyful demeanor.",
            "The image is a close-up photograph of a golden retriever dog. The atmosphere is warm and inviting, with a focus on the dog's joyful expression. The composition centers on the dog's face, capturing its features in sharp detail. The dog has a rich golden-brown coat, with slightly wavy fur that frames its face. Its black nose is prominent, and its tongue is playfully sticking out, suggesting a relaxed and happy demeanor. The dog's eyes are partially visible, adding to its friendly expression. The fur around its face is slightly tousled, indicating movement or playfulness. The background is blurred, using a shallow depth of field to keep the focus on the dog. This background appears to be an outdoor setting, possibly a patio or a backyard, with neutral tones that do not distract from the main subject. The lighting is natural, likely from sunlight, enhancing the warm color tones of the dog's fur. The image has high saturation and brightness, making the colors vivid and the details sharp. The framing is tight, focusing on the dog's head and upper neck, with a slight angle that gives a dynamic perspective. Overall, the image exudes a sense of warmth and happiness, capturing a candid moment of the dog's contentment. There are no discernible texts or additional characters in the image. The composition style is straightforward, emphasizing the dog's expressive features and joyful emotion.",
            "The image is a photograph capturing a lively and playful scene in a sunlit backyard. At the center of the frame, a young border collie is mid-run, with its front paws lifted off the ground and its ears perked up in excitement. Its coat is a striking mix of black and white, with a glossy shine that catches the sunlight. The dog's mouth is slightly open, revealing a pink tongue and a joyful expression that radiates energy. The grass beneath it is a lush, vibrant green, with scattered autumn leaves adding pops of orange and yellow to the scene. In the background, a wooden fence and a few tall trees provide a natural frame, softly blurred to keep the focus on the dog. The lighting is bright yet gentle, with the afternoon sun casting a warm glow across the entire image. The composition style is dynamic, capturing the dog in motion with a slight diagonal tilt, adding a sense of movement and life. The colors are rich and saturated, with sharp focus on the dog's face and fur, while the background blur enhances the feeling of depth. Overall, the image exudes joy, energy, and the carefree spirit of a dog enjoying a perfect day outdoors.",
            "The image is a heartwarming photograph of a small, tan-colored puppy sitting inside a rustic wicker basket. The puppy has large, round eyes that shine with curiosity and a tiny black nose that contrasts with its light fur. Its short, velvety coat is a warm sandy hue, with a faint white patch on its chest. The puppy's ears are slightly floppy, tilting outward in an endearing way. The basket rests on a soft, cream-colored blanket, adding to the cozy feel of the scene. The background is softly lit, with a blurred arrangement of flowers in pastel colors, suggesting an indoor setting near a window. Sunlight filters in gently, creating a warm, golden tone that wraps the entire image in a sense of comfort and calm. The composition is intimate, with the puppy positioned centrally and framed by the curved edges of the basket. The sharp focus on the puppy's eyes and whiskers contrasts with the gentle blur of the surroundings, drawing the viewer's attention directly to its adorable face. Overall, the image captures a perfect blend of innocence, warmth, and charm.",
            "The image is a close-up photograph of a Shiba Inu sitting proudly in a park. Its reddish-brown fur glows warmly under the afternoon sunlight, and its coat appears impeccably groomed. The dog’s almond-shaped eyes are alert yet friendly, and its triangular ears stand upright, giving it a confident appearance. Its mouth is slightly curved into what seems like a natural smile, and its fluffy tail curls neatly over its back. The background consists of a carpet of fallen cherry blossom petals, creating a delicate pink and white texture that contrasts beautifully with the dog’s vibrant coat. A shallow depth of field blurs the distant park benches and trees, keeping the focus entirely on the Shiba Inu. The lighting is bright but soft, enhancing the warmth of the scene without harsh shadows. The composition style is balanced and centered, giving a formal portrait feel while still capturing the relaxed nature of the dog. The colors are vivid, with warm tones dominating and pastel accents adding a gentle harmony. Overall, the photograph conveys a sense of pride, beauty, and serenity.",
            "The image is a photograph of a sleepy beagle resting on a soft plaid blanket. The beagle’s tri-colored coat of white, brown, and black is short but sleek, with the brown patches glowing warmly in the afternoon light. Its long ears drape across the blanket, and its eyes are half-closed in a peaceful, drowsy expression. Its body is curled slightly, with its nose tucked close to its front paws. The blanket is a mix of earthy reds and soft creams, adding to the cozy atmosphere. The background is softly blurred, with hints of a fireplace and a wooden floor, suggesting an indoor setting on a chilly day. The lighting is warm and subdued, creating a soft glow that emphasizes the beagle’s fur texture. The composition style is intimate and comforting, with the dog filling most of the frame. The focus is crisp on the beagle’s face, while the blanket and background fade gently into softness. Overall, the image exudes a sense of calm, warmth, and homey tranquility.",
        ]

    # prompts = [
    #     "The image is a photograph capturing a lively and playful scene in a sunlit backyard. At the center of the frame, a young border collie is mid-run, with its front paws lifted off the ground and its ears perked up in excitement. Its coat is a striking mix of black and white, with a glossy shine that catches the sunlight. The dog's mouth is slightly open, revealing a pink tongue and a joyful expression that radiates energy. The grass beneath it is a lush, vibrant green, with scattered autumn leaves adding pops of orange and yellow to the scene. In the background, a wooden fence and a few tall trees provide a natural frame, softly blurred to keep the focus on the dog. The lighting is bright yet gentle, with the afternoon sun casting a warm glow across the entire image. The composition style is dynamic, capturing the dog in motion with a slight diagonal tilt, adding a sense of movement and life.",
    #     "The image is a heartwarming photograph of a small, tan-colored puppy sitting inside a rustic wicker basket. The puppy has large, round eyes that shine with curiosity and a tiny black nose that contrasts with its light fur. Its short, velvety coat is a warm sandy hue, with a faint white patch on its chest. The puppy's ears are slightly floppy, tilting outward in an endearing way. The basket rests on a soft, cream-colored blanket, adding to the cozy feel of the scene. The background is softly lit, with a blurred arrangement of flowers in pastel colors, suggesting an indoor setting near a window. Sunlight filters in gently, creating a warm, golden tone that wraps the entire image in a sense of comfort and calm. The composition is intimate, with the puppy positioned centrally and framed by the curved edges of the basket.",
    #     "The image is a close-up photograph of a Shiba Inu sitting proudly in a park. Its reddish-brown fur glows warmly under the afternoon sunlight, and its coat appears impeccably groomed. The dog’s almond-shaped eyes are alert yet friendly, and its triangular ears stand upright, giving it a confident appearance. Its mouth is slightly curved into what seems like a natural smile, and its fluffy tail curls neatly over its back. The background consists of a carpet of fallen cherry blossom petals, creating a delicate pink and white texture that contrasts beautifully with the dog’s vibrant coat. A shallow depth of field blurs the distant park benches and trees, keeping the focus entirely on the Shiba Inu. The lighting is bright but soft, enhancing the warmth of the scene without harsh shadows. ",
    #     "The image is a photograph of a sleepy beagle resting on a soft plaid blanket. The beagle’s tri-colored coat of white, brown, and black is short but sleek, with the brown patches glowing warmly in the afternoon light. Its long ears drape across the blanket, and its eyes are half-closed in a peaceful, drowsy expression. Its body is curled slightly, with its nose tucked close to its front paws. The blanket is a mix of earthy reds and soft creams, adding to the cozy atmosphere. The background is softly blurred, with hints of a fireplace and a wooden floor, suggesting an indoor setting on a chilly day. The lighting is warm and subdued, creating a soft glow that emphasizes the beagle’s fur texture. The composition style is intimate and comforting, with the dog filling most of the frame.",
    # ]

    # prompts = [
    #     "The image is a photograph capturing a lively and playful scene in a sunlit backyard. At the center of the frame, a young border collie is mid-run, with its front paws lifted off the ground and its ears perked up in excitement.",
    #     "The image is a heartwarming photograph of a small, tan-colored puppy sitting inside a rustic wicker basket. The puppy has large, round eyes that shine with curiosity and a tiny black nose that contrasts with its light fur.",
    #     "The image is a close-up photograph of a Shiba Inu sitting proudly in a park. Its reddish-brown fur glows warmly under the afternoon sunlight, and its coat appears impeccably groomed.",
    #     "The image is a photograph of a sleepy beagle resting on a soft plaid blanket. The beagle’s tri-colored coat of white, brown, and black is short but sleek, with the brown patches glowing warmly in the afternoon light.",
    # ]

    # prompts = [
    #     'a plane with a wooden handle and a wooden handle\n',
    #     'a collection of comic books on a table\n',
    #     'a brown and white bird standing on gravel\n',
    #     'a green plant with a green stem\n'
    # ]

    caption_embs, emb_masks = t5_model.get_text_embeddings(prompts)

    # import pdb; pdb.set_trace()


    if not args.no_left_padding:
        print(f"processing left-padding...")    
        # a naive way to implement left-padding
        new_emb_masks = torch.flip(emb_masks, dims=[-1])
        new_caption_embs = []
        for idx, (caption_emb, emb_mask) in enumerate(zip(caption_embs, emb_masks)):
            valid_num = int(emb_mask.sum().item())
            print(f'  prompt {idx} token len: {valid_num}')
            new_caption_emb = torch.cat([caption_emb[valid_num:], caption_emb[:valid_num]])
            new_caption_embs.append(new_caption_emb)
        new_caption_embs = torch.stack(new_caption_embs)
    else:
        new_caption_embs, new_emb_masks = caption_embs, emb_masks
    c_indices = new_caption_embs * new_emb_masks[:,:, None]
    c_emb_masks = new_emb_masks

    
    t1 = time.time()
    index_sample = generate(
        gpt_model, c_indices, 1536, 
        c_emb_masks, 
        cfg_scale=args.cfg_scale,
        temperature=args.temperature, top_k=args.top_k,
        top_p=args.top_p, sample_logits=True, 
        )
    sampling_time = time.time() - t1
    print(f"Full sampling takes about {sampling_time:.2f} seconds.")    
    
    t2 = time.time()
    index_sample = torch.flip(index_sample, dims=[1])
    index_sample = index_sample.detach().cpu().numpy()
    samples = vq_model.decoding_with_renderer(index_sample, device)
    decoder_time = time.time() - t2
    print(f"decoder takes about {decoder_time:.2f} seconds.")

    save_image(samples, "sample_{}.png".format(args.gpt_type), nrow=4)
    print(f"image is saved to sample_{args.gpt_type}.png")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--t5-path", type=str, default='pretrained_models/t5-ckpt')
    parser.add_argument("--t5-model-type", type=str, default='flan-t5-xl')
    parser.add_argument("--t5-feature-max-len", type=int, default=256)
    parser.add_argument("--t5-feature-dim", type=int, default=2048)
    parser.add_argument("--no-left-padding", action='store_true', default=False)
    parser.add_argument("--gpt-model", type=str, choices=list(GPT_models.keys()), default="GPT-XL")
    parser.add_argument("--gpt-ckpt", type=str, default=None)
    parser.add_argument("--gpt-type", type=str, choices=['c2i', 't2i'], default="t2i", help="class->image or text->image")  
    parser.add_argument("--cls-token-num", type=int, default=256, help="max token number of condition input")
    parser.add_argument("--precision", type=str, default='bf16', choices=["none", "fp16", "bf16"]) 
    parser.add_argument("--compile", action='store_true', default=False)
    parser.add_argument("--yml-path", type=str, required=True, help="yml for tokenizer")
    parser.add_argument("--vq-ckpt", type=str, required=True, help="ckpt path for vq model")
    parser.add_argument("--sd3-pretrained", type=str, required=True, help="ckpt path for vae")
    parser.add_argument("--codebook-size", type=int, default=16384, help="codebook size for vector quantization")
    parser.add_argument("--codebook-embed-dim", type=int, default=8, help="codebook dimension for vector quantization")
    parser.add_argument("--image-size", type=int, choices=[256, 384, 512], default=512)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=1000, help="top-k value to sample with")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature value to sample with")
    parser.add_argument("--top-p", type=float, default=1.0, help="top-p value to sample with")
    args = parser.parse_args()
    main(args)
