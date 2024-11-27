# Image stitcher

## Prerequsities

Following project is first homework for Robot Control course at MIMUW faculty (2024/25). Below there is presented a
description of how to run each task followed by discussion of results. Let's start with technical information:

- I used `Python 3.11`.
- All dependencies are listed in `requirements.txt` file.

Each task starts after `### Task {i} ###` comment. To run task `i`, you have to uncomment code
between `# --- execution ---` marks between `### Task {i} ###` and `### Task {i+1} ###` and possibly some other
functions if mentioned.

## Task 1

### How to run?

- Uncomment the code between `# --- execution ---` marks. You can select the code with your mouse and
  press `Ctrl + /` if you're using PyCharm or VSC.
- After that, a piece of code between `# ---- Long computations ----` marks will still be commented. Leave it as it is,
  because these computations take a lot of time and results are described below.
- Run the script
- Comment code between `# --- execution ---`

### Discussion of results

After running the script, we display following images:

#### Distorted image

![image](https://github.com/user-attachments/assets/c573dd5f-b98d-452d-bd6a-f2294d9bd33e)

#### Image undistorted using all information (Method 1)

![image](https://github.com/user-attachments/assets/edf4f2c4-7231-4d89-8dd9-790369335650)

#### Image undistorted treating one image as six (Method 2) - optional

![image](https://github.com/user-attachments/assets/61a0efdd-66bc-41f2-8e67-a29d93f3bcb4)

As we can see in the pictures, Method 1 gives better results than Method 2. Contrary to distorted image and Method 2,
straight lines appear straight (highlighted by the red line).

As a result, I used matrix obtained from method 1 to undistort all images.

The script displays also information about reprojection loss:

![image](https://github.com/user-attachments/assets/b52455b7-433d-43b7-a0ca-2e01c51ffc0f)

The loss is higher for Method 1, but it's due to bigger number of images in Method 2.

What's woth noticing is the computation time. `cv2.calibrateCamera` is not linear and takes much more time to compute (
therefore I recommend to keep this method commented). As I found on the internet, this is more or less the plot of time
complexity:

![image](https://github.com/user-attachments/assets/dffcd950-e8c1-4cab-a505-512fc93f53d7)

## Task 2

### How to run it?

- Uncomment code between `# --- execution ---` marks
- Run the script
- Comment code between `# --- execution ---` marks

### Discussion of results

After running the script, following image is displayed:

![image](https://github.com/user-attachments/assets/19cf5332-6897-437f-9bc1-51ea31524e1d)

As one can see, the transformation is performed and no pixels are lost (I used backward homography with nearest
neighbor). In this case, the centre of coordinate system (point (0,0)) is in the top left corner.

## Task 3

### How to run it?

- Uncomment code between `# --- execution ---` marks
- Run the script
- Comment code between `# --- execution ---` marks

### Discussion of results

This time, the script does not display any visible data. It performs a test that:

- picks a random homography (and normalizes it),
- computes the matching pairs based on this homography (random number of points - from 4 to 10 for each
  iteration),
- checks that the implemented method recovers it.

The process repeats 20 times. The homography matrix is recovered with precision up to 1e-14. The test is performed by
following line

`assert np.all(np.isclose(homography_matrix, M, rtol=1e-14, atol=1e-14))`

where `homography_matrix` is input and `M` is a recovered matrix.

## Task 4

### How to run it?

- Uncomment code between `# --- execution ---` marks
- Run the script
- Comment code between `# --- execution ---` marks

# Discussion of results

After choosing corresponding points (between picture 1 and 2), we obtain a following matrix

![image](https://github.com/user-attachments/assets/b1c4f6c0-2924-42fa-8afd-b667e37fd128)

A transformed image looks good as well

![image](https://github.com/user-attachments/assets/54d03b28-d4b5-4edc-ad4a-0b7cbd28b179)

## Task 5

### How to run it

- Uncomment code between `# --- execution ---` marks
- Make cure you have included the `matches` folder
- If you want to highlight the cutting line, uncomment a fragment inside `stitch` function (~line 348)
- Run the script
- Comment code between `# --- execution ---` marks

### Discussion of results

As a result we obtain following image:

![image](https://github.com/user-attachments/assets/d0ae752d-61f9-4cfe-b5fb-a113de6be5e7)

The cutting line is hardly visible, so the algorithm served its purpose.
After uncommenting a fragment in `stitch` function (~line 348), one can see where the line was drawn.

![image](https://github.com/user-attachments/assets/9ef1fda9-524e-4b0f-b8a3-bc9a56b2a2e7)

## Task 6

### How to run it?

- Uncomment code between `# --- execution ---` marks
- Run the script
- Comment code between `# --- execution ---` marks

### Discussion of results

Once again, the results are pretty good. Personally, I can't spot a difference between images generated in task 5 and 6.

![image](https://github.com/user-attachments/assets/ef8d37bc-6b9e-4ffc-9b41-d65cb4c3fe74)

## Task 7

### How to run it?

- Uncomment code between `# --- execution ---` marks
- Run the script
- Comment code between `# --- execution ---` marks

### Discussion of results

The implemented algorithm worked correctly. Below I present a panorama that consists of five images.

![image](https://github.com/user-attachments/assets/e3db4d7b-d845-4f7a-838e-844499f6c405)
