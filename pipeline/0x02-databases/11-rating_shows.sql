-- lists all shows from hbtn_0d_tvshows_rate by their rating
SELECT tv_shows.title, SUM(rate) AS rating FROM tv_shows
INNER JOIN tv_show_ratings
ON tv_show_ratings.show_id = tv_shows.id
GROUP BY title
ORDER BY rating DESC;