import requests
import base64

def test_detect_objects_api(image_path, url, text_prompt, box_threshold, text_threshold):
    # Read the image and encode it
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()

    # Define the payload for the POST request
    payload = {
        "text_prompt": text_prompt,
        "box_threshold": box_threshold,
        "text_threshold": text_threshold
    }
    files = {
        "image": encoded_string
    }

    # Send a POST request to the API
    response = requests.post(url, data=payload, files=files)
    
    # Check the response
    if response.status_code == 200:
        print("Request was successful!")
        print("Response:")
        print(response.json())
    else:
        print(f"Failed to get a successful response, status code: {response.status_code}")
        print("Response:")
        print(response.text)

if __name__ == "__main__":
    # API URL
    url = "http://127.0.0.1:8000/detect/"
    
    # Path to the image you want to test with
    image_path = 'test_images/5b86e93f8654118a.jpg'

    # Parameters
    text_prompt = 'girl'
    box_threshold = 0.3
    text_threshold = 0.0075

    # Call the test function
    test_detect_objects_api(image_path, url, text_prompt, box_threshold, text_threshold)
