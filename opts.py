def saint_plus_opts():
    # Here are all params for saint+ model
    parser = argparse.ArgumentParser()

    # parser.add_argument('--root_path', default='/root/data/ActivityNet', type=str, help='Root directory path of data')

    parser.add_argument('--MAX_SEQ', default=100)
    parser.add_argument('--EMBED_DIMS', default=512)
    parser.add_argument('--ENC_HEADS', default=8)
    parser.add_argument('--DEC_HEADS', default=8)
    parser.add_argument('--NUM_ENCODER', default=4)
    parser.add_argument('--NUM_DECODER', default=4)
    parser.add_argument('--BATCH_SIZE', default=32)
    parser.add_argument(
        '--TRAIN_FILE', default="../input/riiid-test-answer-prediction/train.csv")
    parser.add_argument('--TOTAL_EXE', default=13523)
    parser.add_argument('--TOTAL_CAT', default=10000)

    # device = torch.device("cuda")
    # MAX_SEQ = 100
    # EMBED_DIMS = 512
    # ENC_HEADS = DEC_HEADS = 8
    # NUM_ENCODER = NUM_DECODER = 4
    # BATCH_SIZE = 32
    # TRAIN_FILE = "../input/riiid-test-answer-prediction/train.csv"
    # TOTAL_EXE = 13523
    # TOTAL_CAT = 10000
    args = parser.parse_args()
    return args
