import asyncio
from datetime import datetime

from task._models.custom_content import Attachment
from task._utils.constants import API_KEY, DIAL_URL, DIAL_CHAT_COMPLETIONS_ENDPOINT
from task._utils.bucket_client import DialBucketClient
from task._utils.model_client import DialModelClient
from task._models.message import Message
from task._models.role import Role

class Size:
    """
    The size of the generated image.
    """
    square: str = '1024x1024'
    height_rectangle: str = '1024x1792'
    width_rectangle: str = '1792x1024'


class Style:
    """
    The style of the generated image. Must be one of vivid or natural.
     - Vivid causes the model to lean towards generating hyper-real and dramatic images.
     - Natural causes the model to produce more natural, less hyper-real looking images.
    """
    natural: str = "natural"
    vivid: str = "vivid"


class Quality:
    """
    The quality of the image that will be generated.
     - ‘hd’ creates images with finer details and greater consistency across the image.
    """
    standard: str = "standard"
    hd: str = "hd"

async def _save_images(attachments: list[Attachment]):
    #  1. Create DIAL bucket client
    async with DialBucketClient(base_url=DIAL_URL, api_key=API_KEY) as client:
        #  2. Iterate through Images from attachments, download them and then save here
        for attachment in attachments:
            #  Download image from attachment.url
            image_data = await client.get_file(attachment.url)
            #  Save image locally with attachment.title as file name
            with open(attachment.title, "wb") as image_file:
                image_file.write(image_data)
            #  3. Print confirmation that image has been saved locally
            print(f"Image saved locally: {attachment.title}")


def start() -> None:
    #  1. Create DialModelClient
    client = DialModelClient(
        endpoint=DIAL_CHAT_COMPLETIONS_ENDPOINT,
        deployment_name="dall-e-3",
        api_key=API_KEY
    )
    #  2. Generate image for "Sunny day on Bali"
    message = client.get_completion(
        messages=[
            Message(
                role=Role.USER,
                content="Generate an image of a sunny day on Bali with clear blue skies and palm trees.",
            )
        ],
        custom_fields={
            "quality": Quality.hd,
            "size": Size.width_rectangle,
            "style": Style.vivid,
        }
    )
    #  3. Get attachments from response and save generated message (use method `_save_images`)
    attachments = message.custom_content.attachments
    asyncio.run(_save_images(attachments))
    #  4. Try to configure the picture for output via `custom_fields` parameter.
    #    - Documentation: See `custom_fields`. https://dialx.ai/dial_api#operation/sendChatCompletionRequest
    #  5. Test it with the 'imagegeneration@005' (Google image generation model)

start()
