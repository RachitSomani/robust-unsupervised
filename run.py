from cli import parse_config
import glob

import benchmark
from benchmark import Task, Degradation
from robust_unsupervised import *

config = parse_config()
benchmark.config.resolution = config.resolution

print(config.name)
timestamp = datetime.datetime.now().isoformat(timespec="seconds").replace(":", "")

G = open_generator(config.pkl_path) 
loss_fn = MultiscaleLPIPS()

def run_phase(label: str, variable: Variable, lr: float):        
    # Run optimization loop
    optimizer = NGD(variable.parameters(), lr=lr)
    try:
        for _ in tqdm.tqdm(range(150), desc=label):
            x = variable.to_image()
            loss = loss_fn(degradation.degrade_prediction, x, target).mean()  # Directly compare x to target

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    except KeyboardInterrupt:
        pass

    # Log results
    suffix = "_" + label
    pred = resize_for_logging(variable.to_image(), config.resolution)

    save_image(pred, f"pred{suffix}.png", padding=0)

    save_image(
        torch.cat(
            [
                resize_for_logging(target, config.resolution),
                pred,
            ]
        ),
        f"side_by_side{suffix}.jpg",
        padding=0,
    )
    save_image(
        torch.cat([resize_for_logging(target, config.resolution), pred]),
        f"result{suffix}.jpg",
        padding=0,
    )
    save_image(
        torch.cat([target, pred, (target - pred).abs()]),
        f"fidelity{suffix}.jpg",
        padding=0,
    )

if __name__ == '__main__':
    experiment_path = f"out/{config.name}/{timestamp}/restoration/"
    
    image_paths = sorted(
        [
            os.path.abspath(path)
            for path in (
                glob.glob(config.dataset_path + "/**/*.png", recursive=True)
                + glob.glob(config.dataset_path + "/**/*.jpg", recursive=True)
                + glob.glob(config.dataset_path + "/**/*.jpeg", recursive=True)
                + glob.glob(config.dataset_path + "/**/*.tif", recursive=True)
            )
        ]
    )
    assert len(image_paths) > 0, "No images found!"

    with directory(experiment_path):
        print(experiment_path)
        print(os.path.abspath(config.dataset_path))

        for j, image_path in enumerate(image_paths):
            with directory(f"inversions/{j:04d}"):
                print(f"- {j:04d}")
                
                target = open_image(image_path, config.resolution)  # Load degraded image as target
                save_image(target, f"degraded_input.png")
                
                W_variable = WVariable.sample_from(G)
                run_phase("W", W_variable, config.global_lr_scale * 0.08)

                Wp_variable = WpVariable.from_W(W_variable)
                run_phase("W+", Wp_variable, config.global_lr_scale * 0.02)

                Wpp_variable = WppVariable.from_Wp(Wp_variable)
                run_phase("W++", Wpp_variable, config.global_lr_scale * 0.005)
