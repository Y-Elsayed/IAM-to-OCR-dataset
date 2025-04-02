# IAM-OCR-Pair-Generator

**IAM-OCR-Pair-Generator** is Python toolkit designed to transform full IAM form images into ready-to-use OCR training pairs. The tool automatically segments IAM form images into 
distinct regions—extracting the printed label text from the top and isolating the handwritten paragraph as the input image. This automated pairing creates a dataset of `{image, label}` entries suitable for training and evaluating handwriting recognition models.
