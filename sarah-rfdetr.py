from rfdetr import RFDETRMedium

dataset = "/mnt/swilliams/"

model = RFDETRMedium()

model.train(
    dataset_dir=dataset,
    epochs=20,
    batch_size=16,
    grad_accum_steps=1,
    early_stopping=True,
    early_stopping_patience=3,
    early_stopping_use_ema=True,
    lr=2e-4,
    checkpoint_interval=3
)

print("Training results saved in directory 'output'")

