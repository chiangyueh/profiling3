if __package__:
    from .predictor import predict
else:
    from predictor import predict


INPUTS = {
    "B": 1,
    "N1": 16,
    "N2": 16,
    "S1": 4096,
    "S2": 4096,
    "D": 192,
    "DV": 128,
    "dtype": "BF16",
    "layout": "BNSD",
    "has_drop_mask": 0,
}


def main():
    result = predict(INPUTS)
    print(f"predicted_us={result['predicted_us']:.6f}")


if __name__ == "__main__":
    main()
