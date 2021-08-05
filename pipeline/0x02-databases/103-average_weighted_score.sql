--  creates a stored procedure ComputeAverageWeightedScoreForUser that computes
-- stores the average weighted score for a student
delimiter //

CREATE PROCEDURE ComputeAverageWeightedScoreForUser (IN user_id INT)
BEGIN
UPDATE users
SET average_score=(
    SELECT SUM(score*weight)/SUM(weight) FROM corrections, projects WHERE corrections.user_id = user_id AND corrections.project_id=projects.id
    )
WHERE id=user_id;
END//

delimiter ;