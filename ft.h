#ifndef FT_H
#define FT_H

#include <QSharedDataPointer>

#include <QVector>
#include <QPointF>

class FtData;

class Ft
{
public:
    Ft();
    explicit Ft(int numPnts, double loFreq);
    Ft(const Ft &);
    Ft &operator=(const Ft &);
    ~Ft();

    void setPoint(int i, QPointF pt);
    QPointF &operator[](int i);

    int size() const;
    bool isEmpty() const;
    QPointF at(int i) const;
    double xSpacing() const;
    double minFreq() const;
    double maxFreq() const;
    double loFreq() const;
    double yMin() const;
    double yMax() const;

private:
    QSharedDataPointer<FtData> data;
};

#endif // FT_H
