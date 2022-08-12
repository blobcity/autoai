from blobcity.generative_ai import image as bc

path=r"C:\blobcity\code generator\New folder\samples"

model= bc.train(file=path,epochs=1)

model.generate()

model.get_inter_steps()