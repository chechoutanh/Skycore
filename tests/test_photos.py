from PIL import Image

from skyscore.photos import prepare


def test_prepare_shrinks_and_strips_gps(tmp_path):
    src = tmp_path / "raw"
    src.mkdir()
    exif = Image.Exif()
    exif[0x8825] = {2: (10.0, 46.0, 30.0)}  # GPSInfo: latitude
    exif[0x0112] = 6  # orientation: rotate 90°
    Image.new("RGB", (3000, 2000), "navy").save(src / "night.jpg", exif=exif)

    (out,) = prepare(src, tmp_path / "photos" / "E01")
    with Image.open(out) as im:
        assert len(im.getexif()) == 0
        assert im.size == (1067, 1600)  # rotated upright, longest side capped

    # A second batch is numbered after the first instead of overwriting it.
    (second,) = prepare(src, tmp_path / "photos" / "E01")
    assert (out.name, second.name) == ("01.jpg", "02.jpg")
