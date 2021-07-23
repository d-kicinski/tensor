from pathlib import Path


def clean_text(input_name: str, output_name: str):
    with Path(input_name).open("r") as input_file, Path(output_name).open("w") as output_file:
        for i, line in enumerate(input_file.readlines()):
            line_normalized = line.strip("\t\n\r")
            if line_normalized is None or line_normalized == "":
                continue
            line_normalized = " ".join(line_normalized.split()).lower()
            line_normalized = line_normalized if i == 0 else " " + line_normalized
            if line_normalized:
                output_file.write(line_normalized)


if __name__ == '__main__':
    clean_text("pan-tadeusz.txt", "pan-tadeusz-clean.txt")
