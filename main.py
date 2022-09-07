import os
import cv2
from proc import GraffitiProcessor
from configs import *
import random
random.seed(42)

if __name__ == "__main__":
    used_gr = []
    proc = GraffitiProcessor(path_to_data, path_to_gr, colors)

    for gost in proc.path_2_gosts:
        os.makedirs(os.path.join(path_to_data, gost, "synthetics"), exist_ok=True)
        clean_amount = len(os.listdir(os.path.join(path_to_data, gost, "clean")))
        dirty_amount = len(os.listdir(os.path.join(path_to_data, gost, "graffiti"))) + len(
            os.listdir(os.path.join(path_to_data, gost, "synthetics")))
        print(">>> GOST: {} |  dirty_amount: {} | clean_amount: {}".format(gost, dirty_amount, clean_amount))

        images_path = [os.path.join(path_to_data, gost, "clean", i) for i in
                       os.listdir(os.path.join(path_to_data, gost, "clean"))]

        while dirty_amount != 1000:
            rand_gr_path, gr_type = proc.choose_rand_gr()
            if rand_gr_path in used_gr:
                continue

            random_image_path = random.choice(images_path)
            if os.path.basename(random_image_path)[:-4] + "_SYN.jpg" in os.listdir(
                    os.path.join(path_to_data, gost, "synthetics")):
                continue

            graffiti_image = cv2.imread(rand_gr_path)
            image = cv2.imread(random_image_path)
            im_show = image.copy()

            gh, gw, _ = graffiti_image.shape
            ih, iw, _ = image.shape

            kwargs = {"sign_image": image, "graffiti":graffiti_image}
            kwargs['left'], kwargs['top'], kwargs['right'], kwargs['bottom'] = 5, 5, iw - 5, ih - 5
            if gr_type == "graffiti":
                kwargs['min_gr_size'], kwargs['max_gr_size'] = int(((iw + ih) / 2) * 0.5), int(((iw + ih) / 2) * 0.7)
            elif gr_type == "nakleiki":
                kwargs['min_gr_size'], kwargs['max_gr_size'] = int(((iw + ih) / 2) * 0.2), int(((iw + ih) / 2) * 0.25)

            try:
                new_image = proc.apply_single_transform(gr_type, kwargs)
            except Exception:
                continue

            im_show = cv2.resize(im_show, (100, 100))
            cv2.imshow("orig", im_show)
            cv2.imshow("transformed", image)
            wk = cv2.waitKey(0) & 0xFF

            # SAVE IMAGE
            if wk == ord("s"):
                image_name = os.path.basename(random_image_path)[:-4] + "_SYN.jpg"
                save_path = os.path.join(path_to_data, gost, "synthetics", image_name)
                cv2.imwrite(save_path, image)
                dirty_amount += 1
                used_gr.append(rand_gr_path)
                print("graffiti_amount now is: {}".format(dirty_amount), end="\r")

            # CHANGE GRAFFITI COLOR
            if wk == ord("c"):
                proc.color = next(proc.colors_cycle)

            # EXIT
            if wk == ord("e"):
                exit()

            # NEXT GOST
            if wk == ord("n"):
                break