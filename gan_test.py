from blobcity.generative_ai import image as bc

path=r"C:\blobcity\code generator\New folder\samples"

model= bc.train(file=path,epochs=5)

model.generate()

model.generate_inter_steps()