import pandas as pd


def aggregate():
    with open('outputs/same_shape.log') as f:
        results = [x.strip() for x in f.readlines()]

    pairs = []
    for r in results:
        n1, n2 = int(r.split()[0]), int(r.split()[2])
        pairs.append((n1, n2))

    groups = []
    for pair in pairs:
        added = False
        for group in groups:
            if pair[0] in group:
                group.add(pair[1])
                added = True
        if not added:
            s = set()
            s.add(pair[0])
            s.add(pair[1])
            groups.append(s)

    df = []
    for group in groups:
        df.append((len(group), sorted(list(group))))
    df = pd.DataFrame(df, columns=['Num elems', 'elements'])

    df.sort_values(by='Num elems', ascending=False).to_csv('outputs/shape.csv', index=False)


if __name__ == '__main__':
    aggregate()
