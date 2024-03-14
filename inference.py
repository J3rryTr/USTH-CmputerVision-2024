import torch

model = torch.hub.load('private/path/', 'custom', path='output/best', force_reload=True, source='local')

# Changing settings to prevent finding the faces multiple times
model.conf = 0.5
model.iou = 0.3

output = model('insert/imgs')
output.show()