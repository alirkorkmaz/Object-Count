def orientation(p, q, r):
    """
    Üç noktanın yönünü belirler.
    Döndürür:
    0 -> kolinear
    1 -> saat yönü
    2 -> saat yönü tersi
    """
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if val == 0:
        return 0
    return 1 if val > 0 else 2

def on_segment(p, q, r):
    """
    q noktası, p ve r noktaları arasında segment üzerindeyse True döner.
    """
    if (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
        q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1])):
        return True
    return False

def check_line_crossing(point1, point2, line_p1, line_p2):
    """
    Bir nesnenin (point1 -> point2) çizgiyi (line_p1 -> line_p2) kesip kesmediğini kontrol eder.
    """
    o1 = orientation(line_p1, line_p2, point1)
    o2 = orientation(line_p1, line_p2, point2)
    o3 = orientation(point1, point2, line_p1)
    o4 = orientation(point1, point2, line_p2)

    # Genel durumda kesişme
    if o1 != o2 and o3 != o4:
        return True

    # Özel durumlar: kolinear kesişmeler
    if o1 == 0 and on_segment(line_p1, point1, line_p2):
        return True
    if o2 == 0 and on_segment(line_p1, point2, line_p2):
        return True
    if o3 == 0 and on_segment(point1, line_p1, point2):
        return True
    if o4 == 0 and on_segment(point1, line_p2, point2):
        return True

    return False
