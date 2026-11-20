# Requirements Document

## Introduction

This document outlines the requirements for developing an advanced Flask-based web interface for the deepfake detection system. The web interface will allow users to upload images through a browser and receive real-time predictions about whether the images are real or AI-generated/fake, leveraging the existing hybrid CNN-Transformer model.

## Glossary

- **Flask_Application**: The web server application built using the Flask framework that serves the user interface and handles HTTP requests
- **Deepfake_Model**: The pre-trained hybrid CNN-Transformer neural network model stored in 'best_hybrid_model.pth' that classifies images as real or fake
- **Upload_Handler**: The Flask route component that processes image file uploads from users
- **Prediction_Engine**: The backend component that loads images, applies transformations, and invokes the Deepfake_Model for inference
- **Web_Interface**: The HTML/CSS/JavaScript frontend that users interact with in their browser
- **Result_Display**: The visual component showing prediction results including classification, confidence scores, and probability distributions
- **Image_Preprocessor**: The component that applies necessary transformations (resize, normalize) to uploaded images before model inference

## Requirements

### Requirement 1

**User Story:** As a user, I want to upload an image through a web browser, so that I can check if it's real or fake without installing any desktop software

#### Acceptance Criteria

1. WHEN a user accesses the web application, THE Flask_Application SHALL display a responsive web page with an image upload interface
2. WHEN a user selects an image file, THE Web_Interface SHALL validate that the file is an image format (JPEG, PNG, BMP, TIFF)
3. WHEN a user submits a valid image, THE Upload_Handler SHALL accept the file and store it temporarily for processing
4. IF a user submits an invalid file type, THEN THE Flask_Application SHALL display an error message indicating acceptable file formats
5. THE Web_Interface SHALL provide visual feedback during the upload process with a loading indicator

### Requirement 2

**User Story:** As a user, I want to see the prediction results with confidence scores, so that I can understand how certain the model is about its classification

#### Acceptance Criteria

1. WHEN an image is uploaded, THE Prediction_Engine SHALL load the Deepfake_Model and process the image within 5 seconds
2. WHEN the Prediction_Engine completes inference, THE Flask_Application SHALL return a JSON response containing the predicted class, confidence score, and probability distribution
3. THE Result_Display SHALL show the uploaded image alongside the prediction results
4. THE Result_Display SHALL display the classification (REAL or FAKE) with color coding (green for real, red for fake)
5. THE Result_Display SHALL show both real and fake probabilities as percentage values and visual progress bars

### Requirement 3

**User Story:** As a user, I want to see a visual representation of the prediction probabilities, so that I can quickly understand the model's confidence distribution

#### Acceptance Criteria

1. WHEN prediction results are available, THE Result_Display SHALL render two horizontal progress bars representing real and fake probabilities
2. THE Result_Display SHALL color-code the progress bars (green for real probability, red for fake probability)
3. THE Result_Display SHALL display percentage values on or near each progress bar
4. THE Web_Interface SHALL update the display smoothly without page refresh using AJAX
5. THE Result_Display SHALL show an emoji indicator (✅ for real, ⚠️ for fake) alongside the prediction

### Requirement 4

**User Story:** As a user, I want to upload multiple images in succession, so that I can test several images without restarting the application

#### Acceptance Criteria

1. WHEN a prediction is complete, THE Web_Interface SHALL allow the user to upload another image immediately
2. THE Flask_Application SHALL clear previous results before displaying new predictions
3. THE Flask_Application SHALL maintain the Deepfake_Model in memory between requests to ensure fast subsequent predictions
4. THE Upload_Handler SHALL handle concurrent requests from multiple users without conflicts
5. THE Flask_Application SHALL clean up temporary uploaded files after processing to prevent disk space issues

### Requirement 5

**User Story:** As a user, I want the web interface to work on different devices and screen sizes, so that I can use it on desktop, tablet, or mobile

#### Acceptance Criteria

1. THE Web_Interface SHALL use responsive CSS design that adapts to screen widths from 320px to 1920px
2. WHEN accessed on mobile devices, THE Web_Interface SHALL display elements in a single column layout
3. WHEN accessed on desktop devices, THE Web_Interface SHALL display elements in an optimized multi-column layout where appropriate
4. THE Web_Interface SHALL ensure all interactive elements (buttons, upload areas) are touch-friendly with minimum 44px touch targets
5. THE Web_Interface SHALL maintain readability with appropriate font sizes across all device types

### Requirement 6

**User Story:** As a developer, I want the Flask application to handle errors gracefully, so that users receive helpful feedback when something goes wrong

#### Acceptance Criteria

1. IF the Deepfake_Model file is missing, THEN THE Flask_Application SHALL display an error message indicating the model needs to be trained first
2. IF an image processing error occurs, THEN THE Flask_Application SHALL return a user-friendly error message without exposing technical details
3. IF the server encounters an internal error, THEN THE Flask_Application SHALL log the error details and return a generic error message to the user
4. THE Flask_Application SHALL validate image file size and reject files larger than 10MB with an appropriate error message
5. THE Flask_Application SHALL handle GPU/CPU device selection automatically and log the device being used

### Requirement 7

**User Story:** As a user, I want to see additional details about the prediction, so that I can better understand the analysis

#### Acceptance Criteria

1. THE Result_Display SHALL show the original uploaded image with appropriate sizing
2. THE Result_Display SHALL display the prediction timestamp
3. THE Result_Display SHALL show the confidence score as a percentage with two decimal places
4. THE Web_Interface SHALL provide a download option for the prediction results
5. THE Result_Display SHALL include a visual indicator showing the processing time for the prediction

### Requirement 8

**User Story:** As a developer, I want the Flask application to be easy to deploy and configure, so that it can be set up quickly in different environments

#### Acceptance Criteria

1. THE Flask_Application SHALL load configuration from environment variables or a configuration file
2. THE Flask_Application SHALL provide clear console output indicating the server status, port, and device (CPU/GPU) being used
3. THE Flask_Application SHALL include a requirements.txt file listing all Python dependencies
4. THE Flask_Application SHALL run on both Windows and Linux operating systems without code modifications
5. THE Flask_Application SHALL provide a simple command to start the server (e.g., python app.py)
