#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


# Load the image
image = cv2.imread('img.jpg')


# In[3]:


# Display the original image
plt.figure(figsize=(6, 6))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Original Image'), plt.axis('off')
plt.show()


# In[4]:


# Image Sampling
def image_sampling(img, factor):
    sampled_img = img[::factor, ::factor]
    return sampled_img


# In[5]:


# Image Quantization
def image_quantization(img, levels):
    quantized_img = np.floor_divide(img, 256 // levels) * (256 // levels)
    return quantized_img


# In[6]:


# Sampling the image by a factor of 2
sampled_image = image_sampling(image, 2)


# In[7]:


# Quantizing the image to 4 levels
quantized_image = image_quantization(image, 4)


# In[8]:


# Display the sampled image
plt.figure(figsize=(6, 6))
plt.imshow(cv2.cvtColor(sampled_image, cv2.COLOR_BGR2RGB))
plt.title('Sampled Image (Factor: 2)')
plt.axis('off')
plt.show()

# Display the quantized image
plt.figure(figsize=(6, 6))
plt.imshow(cv2.cvtColor(quantized_image, cv2.COLOR_BGR2RGB))
plt.title('Quantized Image (Levels: 4)')
plt.axis('off')
plt.show()


# In[ ]:




