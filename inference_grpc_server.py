import inference.grpc_service as gs

def main():
    gs.InferenceServiceImpl.serve()


if __name__ == '__main__':
    main()
