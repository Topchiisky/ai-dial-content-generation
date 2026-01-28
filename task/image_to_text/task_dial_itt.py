import asyncio
from io import BytesIO
from pathlib import Path

from task._models.custom_content import Attachment, CustomContent
from task._utils.constants import API_KEY, DIAL_URL, DIAL_CHAT_COMPLETIONS_ENDPOINT
from task._utils.bucket_client import DialBucketClient
from task._utils.model_client import DialModelClient
from task._models.message import Message
from task._models.role import Role


async def _put_image() -> Attachment:
    file_name = 'dialx-banner.png'
    image_path = Path(__file__).parent.parent.parent / file_name
    mime_type_png = 'image/png'
    #  1. Create DialBucketClient and open async context
    async with DialBucketClient(base_url=DIAL_URL, api_key=API_KEY) as client:
        #  2. Open image file
        with open(image_path, "rb") as image_file:
            image_bytes = image_file.read()
        #  3. Use BytesIO to load bytes of image
        image_stream = BytesIO(image_bytes)
        #  4. Upload file with client
        bucket_file = await client.put_file(name=file_name, mime_type=mime_type_png, content=image_stream)
        #  5. Return Attachment object with title (file name), url and type (mime type)
        return Attachment(
            title=file_name,
            url=bucket_file.get("url"),
            type=mime_type_png
        )


def start() -> None:
    #  1. Create DialModelClient
    client = DialModelClient(
        endpoint=DIAL_CHAT_COMPLETIONS_ENDPOINT,
        deployment_name="gpt-4o",
        api_key=API_KEY
    )
    #  2. Upload image (use `_put_image` method )
    attachment = asyncio.run(_put_image())

    #  3. Print attachment to see result
    print(attachment)

    #  4. Call chat completion via client with list containing one Message:
    #    - role: Role.USER
    #    - content: "What do you see on this picture?"
    #    - custom_content: CustomContent(attachments=[attachment])
    message = client.get_completion(
        messages=[
            Message(
                role=Role.USER,
                content="What do you see on this picture?",
                custom_content=CustomContent(attachments=[attachment])
            )
        ]
    )

    print("Response from model:")
    print(message)
    #  ---------------------------------------------------------------------------------------------------------------
    #  Note: This approach uploads the image to DIAL bucket and references it via attachment. The key benefit of this
    #        approach that we can use Models from different vendors (OpenAI, Google, Anthropic). The DIAL Core
    #        adapts this attachment to Message content in appropriate format for Model.
    #  TRY THIS APPROACH WITH DIFFERENT MODELS!
    #  Optional: Try upload 2+ pictures for analysis


start()
