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

    void setPoint(int i, QPointF pt, double ignoreRange = 0.0);
    void resize(int n, double ignoreRange = 0.0);
    QPointF &operator[](int i);
    void reserve(int n);
    void append(QPointF pt, double ignoreRange = 0.0);
    void trim(double minOffset, double maxOffset);

    int size() const;
    bool isEmpty() const;
    QPointF at(int i) const;
    double xSpacing() const;
    double minFreq() const;
    double maxFreq() const;
    double loFreq() const;
    double yMin() const;
    double yMax() const;
    QVector<double> xData() const;
    QVector<double> yData() const;
    QVector<QPointF> toVector() const;

private:
    QSharedDataPointer<FtData> data;
};

#endif // FT_H
