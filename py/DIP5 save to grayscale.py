#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2


# In[2]:


# Read the grayscale image
image = cv2.imread('img.jpg', cv2.IMREAD_GRAYSCALE)


# In[3]:


if image is not None:
    # Display the grayscale image
    cv2.imshow('Grayscale Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the grayscale image
    cv2.imwrite('img2GS.jpg', image)
    print("Grayscale image saved successfully.")
else:
    print("Error: Unable to read the image. Please check the file path.")


# In[ ]:




