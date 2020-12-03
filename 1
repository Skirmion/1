#include <string>
#include <SFML/Graphics.hpp>
#include <iostream>
#include <Cassert>
#include "windows.h"
#include <cmath> 


using namespace std;

struct Vector2f
{
	float x;
	float y;

	Vector2f()
	{
		x = 0;
		y = 0;
	}

	Vector2f(float x, float y)
	{
		this->x = x;
		this->y = y;
	}

	Vector2f operator+ (Vector2f v)
	{
		return Vector2f(x + v.x, y + v.y);
	}

	Vector2f operator- (Vector2f v)
	{
		return Vector2f(x - v.x, y - v.y);
	}

	Vector2f operator* (float n)
	{
		return Vector2f(n * x, n * y);
	}

	Vector2f operator/ (float n)
	{
		return Vector2f(x / n, y / n);
	}

	float scalar(Vector2f v)
	{
		return  x * v.x + y * v.y;
	}

	float length()
	{
		return sqrt(x * x + y * y);
	}

	Vector2f normalize()
	{
		float l = this->length();
		return Vector2f(x / l, y / l);
	}

	float angle(Vector2f u)
	{
		return((atan((u.y - y) / (u.x - x))) * 180 / 3.14);
	}

};


struct Sphere
{
public:

	Vector2f position;
	Vector2f velocity;
	Vector2f acceleration;
	int radius;
	int red;
	int green;
	int blue;
	int m;

	Sphere()
	{
		position = { 0, 0 };
		velocity = { 0, 0 };
		acceleration = { 0, 0 };
		radius = 0;
		red = 0;
		green = 0;
		blue = 0;
		m = 0;
	}

	Sphere(Vector2f position, Vector2f velocity, Vector2f acceleration, int radius, int red, int green, int blue, int m)
	{
		this->position = position;
		this->velocity = velocity;
		this->acceleration = acceleration;
		this->radius = radius;
		this->red = red;
		this->green = green;
		this->blue = blue;
		this->m = m;
	}


	void drawSphere(sf::RenderWindow* myWindow)
	{
		int numberofCircles = 10;
		float x_centerofCircles = position.x;
		float y_centerofCircles = position.y;
		sf::CircleShape circle(0, 30);
		for (int i = 0; i < numberofCircles; i++)
		{
			circle.setRadius(radius - i * radius / numberofCircles);
			circle.setFillColor(sf::Color(i * red / numberofCircles, i * green / numberofCircles, i * blue / numberofCircles));
			circle.setPosition(position.x + (position.x - x_centerofCircles) * i / numberofCircles - radius + radius * i / numberofCircles, position.y + (position.y - y_centerofCircles) * i / numberofCircles - radius + radius * i / numberofCircles);
			myWindow->draw(circle);
		}
	}

	void moveSphere(float t)
	{
		position = position + velocity * t;
	}

	void collideSphereWall(int windowX, int windowY, int t)
	{
		if (position.x + radius + velocity.x * t > windowX)
		{
			velocity.x = -abs(velocity.x);
			position.x = position.x + 2 * (windowX - position.x) - 2 * radius;
		}

		if (position.x + velocity.x * t - radius < 0)
		{
			velocity.x = abs(velocity.x);
			position.x = -position.x + 2 * radius;
		}
		if (position.y + radius + velocity.y * t > windowY)
		{
			velocity.y = -abs(velocity.y);
			position.y = position.y + 2 * (windowY - position.y) - 2 * radius;
		}
		if (position.y + velocity.y * t - radius < 0)
		{
			velocity.y = abs(velocity.y);
			position.y = -position.y + 2 * radius;
		}
	}

	bool checkCollisionTwoSphers(Sphere sphere2)
	{
		return (pow(sphere2.position.x - position.x, 2) + pow(sphere2.position.y - position.y, 2) < pow(radius + sphere2.radius, 2));
	}

	void collideTwoSphers(Sphere * sphere2, int t, int windowX, int windowY)
	{
		float dv1 = velocity.x;
		float du1 = velocity.y;
		float dv2 = sphere2->velocity.x;
		float du2 = sphere2->velocity.y;

		float x1 = position.x;
		float y1 = position.y;
		float x2 = sphere2->position.x;
		float y2 = sphere2->position.y;

		float dist = sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2));
		position.x = position.x + (radius + sphere2->radius - dist) * (x1 - x2) / (2 * dist);
		position.y = position.y + (radius + sphere2->radius - dist) * (y1 - y2) / (2 * dist);
		sphere2->position.x = sphere2->position.x + (radius + sphere2->radius - dist) * (x2 - x1) / (2 * dist);
		sphere2->position.y = sphere2->position.y + (radius + sphere2->radius - dist) * (y2 - y1) / (2 * dist);

		if ((x1 + radius + velocity.x * t > windowX) or (x1 + velocity.x * t - radius < 0))
		{
			sphere2->position.x = sphere2->position.x + (radius + sphere2->radius - dist) * (x2 - x1) / dist;
		}

		if ((x2 + sphere2->radius + sphere2->velocity.x * t > windowX) or (x2 + velocity.x * t - radius < 0))
		{
			position.x = position.x + (radius + sphere2->radius - dist) * (x1 - x2) / dist;
			sphere2->position.x = x2;
		}
		if ((y1 + radius + velocity.y * t > windowY) or (y1 + velocity.y * t - radius < 0))
		{
			sphere2->position.y = sphere2->position.y + (radius + sphere2->radius - dist) * (y2 - y1) / dist;
		}

		if ((y2 + sphere2->radius + sphere2->velocity.y * t > windowY) or (y2 + velocity.y * t - radius < 0))
		{
			position.y = position.y + (radius + sphere2->radius - dist) * (y1 - y2) / dist;
			sphere2->position.y = y2;
		}

		velocity.x = (2 * sphere2->m * dv2 + (m - sphere2->m) * dv1) / (m + sphere2->m);
		velocity.y = (2 * sphere2->m * du2 + (m - sphere2->m) * du1) / (m + sphere2->m);
		sphere2->velocity.x = (2 * m * dv1 + (sphere2->m - m) * dv2) / (m + sphere2->m);
		sphere2->velocity.y = (2 * m * du1 + (sphere2->m - m) * du2) / (m + sphere2->m);

		assert(dist);
	}

	void changeVelocity(float t)
	{
		velocity = velocity + acceleration * t;
	}

	bool ballsDangerouslyClose(float x, float y)
	{
		if (pow((x - position.x), 2) + pow((y - position.y), 2) <= (pow(25, 2)))
		{
			return true;
		};
		return false;
	};
};



class Game
{
public:
	Game();
	void run();

private:
	void ProcessEvents();
	void update();
	void Map();
	void render();
	void handlePlayerInput(sf::Keyboard::Key key, bool isPressed);
	void gameOver(int one);
	void InWindowtank();
private:
	sf::RenderWindow myWindow;
	sf::CircleShape myPlayer1; sf::CircleShape myPlayer2;
	bool movingLeft1 = 0, movingUp1 = 0, movingRight1 = 0, movingDown1 = 0, shot1 = 0;
	bool movingLeft2 = 0, movingUp2 = 0, movingRight2 = 0, movingDown2 = 0, shot2 = 0;

	Sphere spheres1[30]; Sphere spheres2[30];

	int i = 0; int g = 0; float t = 0.1; int gv = 1;

	string direction1 = { 0,0 }; string direction2 = { 0,0 };

};

Game::Game() : myWindow(sf::VideoMode(1920, 1080), "myWindow"), myPlayer1(), myPlayer2()
{
	myPlayer1.setRadius(15.f);
	myPlayer1.setFillColor(sf::Color::Red);
	myPlayer1.setPosition(100.f, 100.f);

	myPlayer2.setRadius(15.f);
	myPlayer2.setFillColor(sf::Color::Blue);
	myPlayer2.setPosition(1820.f, 980.f);

	for (int s = 0; s < 30; s++)
	{
		spheres1[s] = { {-100.f, -100.f} , {0.f, 0.f},  {0, 0}, 10, 0, 0, 255, 10 };
		spheres2[s] = { {-100.f, -100.f} , {0.f, 0.f},  {0, 0}, 10, 0, 0, 255, 10 };
	};


}

void Game::run()
{

	while (myWindow.isOpen())
	{
		ProcessEvents();
		if (gv)
		{
			Map();
			update();
			render();
		};
	}
}

void Game::ProcessEvents()
{
	sf::Event event;
	while (myWindow.pollEvent(event))
	{
		switch (event.type)
		{
		case sf::Event::KeyPressed: handlePlayerInput(event.key.code, true); break;
		case sf::Event::KeyReleased: handlePlayerInput(event.key.code, false); break;
		case sf::Event::Closed: myWindow.close(); break;
		}
	}
}
void Game::Map()
{
	myWindow.clear();
	for (int d = 0; d < 3; d++)
	{
		for (int k = 0; k < 3; k++)
		{

			sf::Vertex line1[] =
			{
				sf::Vertex(sf::Vector2f(640 * d + 1.f, 360 * k - 90.f)),
				sf::Vertex(sf::Vector2f(640 * d + 1.f, 360 * k + 90.f))
			};
			myWindow.draw(line1, 2, sf::Lines);


			sf::Vertex line2[] =
			{
				sf::Vertex(sf::Vector2f(640 * d - 160.f, 360 * k + 1.f)),
				sf::Vertex(sf::Vector2f(640 * d + 160.f, 360 * k + 1.f))
			};
			myWindow.draw(line2, 2, sf::Lines);


			sf::Vertex line3[] =
			{
				sf::Vertex(sf::Vector2f(640 * d + 320.f, 360 * k + 90.f)),
				sf::Vertex(sf::Vector2f(640 * d + 320.f, 360 * k + 270.f))
			};
			myWindow.draw(line3, 2, sf::Lines);


			sf::Vertex line4[] =
			{
				sf::Vertex(sf::Vector2f(640 * d + 480.f, 360 * k - 1.f + 360.f)),
				sf::Vertex(sf::Vector2f(640 * d + 800.f, 360 * k - 1.f + 360.f))
			};
			myWindow.draw(line4, 2, sf::Lines);


			sf::Vertex line5[] =
			{
				sf::Vertex(sf::Vector2f(640 * d - 1.f + 640.f, 360 * k + 270.f)),
				sf::Vertex(sf::Vector2f(640 * d - 1.f + 640.f, 360 * k + 450.f))
			};
			myWindow.draw(line5, 2, sf::Lines);

		};
	};
	sf::Vertex lineX1[] =
	{
		sf::Vertex(sf::Vector2f(1.f, 1.f)),
		sf::Vertex(sf::Vector2f(1919.f, 1.f))
	};
	myWindow.draw(lineX1, 2, sf::Lines);


	sf::Vertex lineX2[] =
	{
		sf::Vertex(sf::Vector2f(1.f, 1079.f)),
		sf::Vertex(sf::Vector2f(1919.f, 1079.f))
	};
	myWindow.draw(lineX2, 2, sf::Lines);


	sf::Vertex lineY1[] =
	{
		sf::Vertex(sf::Vector2f(1.f, 1.f)),
		sf::Vertex(sf::Vector2f(1.f, 1079))
	};
	myWindow.draw(lineY1, 2, sf::Lines);


	sf::Vertex lineY2[] =
	{
		sf::Vertex(sf::Vector2f(1919.f, 1.f)),
		sf::Vertex(sf::Vector2f(1919.f, 1079.f))
	};
	myWindow.draw(lineY2, 2, sf::Lines);


}

void Game::handlePlayerInput(sf::Keyboard::Key key, bool isPressed)
{
	if (key == sf::Keyboard::W or key == sf::Keyboard::S or key == sf::Keyboard::A or key == sf::Keyboard::D) direction1 = { 0,0 };
	if (key == sf::Keyboard::W) movingUp1 = isPressed, direction1[1] = -1;
	else if (key == sf::Keyboard::S) movingDown1 = isPressed, direction1[1] = 1;
	else if (key == sf::Keyboard::A) movingLeft1 = isPressed, direction1[0] = -1;
	else if (key == sf::Keyboard::D) movingRight1 = isPressed, direction1[0] = 1;
	else if (key == sf::Keyboard::X) shot1 = isPressed;

	if (key == sf::Keyboard::Up or key == sf::Keyboard::Down or key == sf::Keyboard::Left or key == sf::Keyboard::Right) direction2 = { 0,0 };
	if (key == sf::Keyboard::Up) movingUp2 = isPressed, direction2[1] = -1;
	else if (key == sf::Keyboard::Down) movingDown2 = isPressed, direction2[1] = 1;
	else if (key == sf::Keyboard::Left) movingLeft2 = isPressed, direction2[0] = -1;
	else if (key == sf::Keyboard::Right) movingRight2 = isPressed, direction2[0] = 1;
	else if (key == sf::Keyboard::RShift) shot2 = isPressed;
	if (gv == 0)
	{
		if (key == sf::Keyboard::F) gv = 1;

		for (int s = 0; s < 30; s++)
		{
			spheres1[s] = { {-1.f, -1.f} , {0.f, 0.f},  {0, 0}, 10, 0, 0, 255, 10 };
			spheres2[s] = { {-1.f, -1.f} , {0.f, 0.f},  {0, 0}, 10, 0, 0, 255, 10 };
		};

		myPlayer1.setPosition(100.f, 100.f);
		myPlayer2.setPosition(1820.f, 980.f);
	};


}



void Game::update()
{
	

	sf::Vector2f movement1(0.f, 0.f);
	sf::Vector2f pos1(0.f, 0.f);
	if (movingUp1) movement1.y -= 5.f;
	if (movingDown1) movement1.y += 5.f;
	if (movingLeft1) movement1.x -= 5.f;
	if (movingRight1) movement1.x += 5.f;
	if (shot1 and (direction1[0] != 0 or direction1[1] != 0)) { spheres1[i] = Sphere(), pos1 = myPlayer1.getPosition(); spheres1[i] = { {pos1.x + myPlayer1.getRadius(), pos1.y + myPlayer1.getRadius()} , {float(300 * direction1[0]), float(300 * direction1[1])},  {0, 0}, 10, 255, 0, 0, 10 }; i += 1; };
	if (i == 30) i = 0;

	sf::Vector2f movement2(0.f, 0.f);
	sf::Vector2f pos2(0.f, 0.f);
	if (movingUp2) movement2.y -= 5.f;
	if (movingDown2) movement2.y += 5.f;
	if (movingLeft2) movement2.x -= 5.f;
	if (movingRight2) movement2.x += 5.f;
	if (shot2 and (direction2[0] != 0 or direction2[1] != 0)) { spheres2[g] = Sphere(), pos2 = myPlayer2.getPosition(); spheres2[g] = { {pos2.x + myPlayer2.getRadius(), pos2.y + myPlayer2.getRadius()} , {float(300 * direction2[0]), float(300 * direction2[1])},  {0, 0}, 10, 0, 0, 255, 10 }; g += 1; };
	if (g == 30) g = 0;

    InWindowtank();
	myPlayer1.move(movement1);
	myPlayer2.move(movement2);
}



void Game::gameOver(int one)
{
	myWindow.clear();
	sf::Font font;
	font.loadFromFile("CyrilicOld.ttf");
	sf::Text text("Win", font, 30);
	text.setFillColor(sf::Color::Red);
	text.setPosition(900, 300);
	text.setStyle(sf::Text::Bold);

	if (one == 1)
	{
		text.setString("         Win 1 \nPress F to pay respect");
	};

	if (one == 0)
	{
		text.setString("         Win 2 \nPress F to pay respect");
	};

	for (int h = 0; h < 30; h++)
	{
		spheres1[h].velocity = { 0,0 };
		spheres2[h].velocity = { 0,0 };
	}
	myWindow.draw(text);
	myWindow.display();
	gv = 0;

};


void Game::render()
{

	for (int j = 0; j < 30; j++)
	{

		spheres1[j].moveSphere(t);
		spheres1[j].drawSphere(&myWindow);

		sf::Vector2f poss2(0.f, 0.f);
		poss2 = myPlayer2.getPosition();
		if (spheres1[j].ballsDangerouslyClose(poss2.x, poss2.y)) { gameOver(1); };

		spheres2[j].moveSphere(t);
		spheres2[j].drawSphere(&myWindow);

		sf::Vector2f poss1(0.f, 0.f);
		poss1 = myPlayer1.getPosition();
		if (spheres2[j].ballsDangerouslyClose(poss1.x, poss1.y)) { gameOver(0); };

	};
	if (gv)
	{
		myWindow.draw(myPlayer1);
		myWindow.draw(myPlayer2);
		myWindow.display();
	};


}

void Game::InWindowtank()
{
	sf::Vector2f poss1(0.f, 0.f);
	poss1 = myPlayer1.getPosition();
	sf::Vector2f poss2(0.f, 0.f);
	poss2 = myPlayer2.getPosition();
	if (poss1.x < 0)
	{
		poss1.x = 0;
		myPlayer1.setPosition(poss1.x, poss1.y);
	}
	if (poss1.x + 2*myPlayer1.getRadius() > 1920)
	{
		poss1.x = 1920 - 2*myPlayer1.getRadius();
		myPlayer1.setPosition(poss1.x, poss1.y);
	}
	if (poss1.y < 0)
	{
		poss1.y = 0;
		myPlayer1.setPosition(poss1.x, poss1.y);
	}

	if (poss1.y + 2*myPlayer1.getRadius() > 1080)
	{
		poss1.y = 1080 - 2*myPlayer1.getRadius();
		myPlayer1.setPosition(poss1.x, poss1.y);
	}
}



int main()
{
	Game game;
	game.run();
	return 0;
}
